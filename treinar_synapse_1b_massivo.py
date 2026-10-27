#!/usr/bin/env python3
"""
MOTOR DE PRÉ-TREINAMENTO MASSIVO DO SYNAPSE 1B (MoE 1.2B)
Desenvolvido por Nandinho.
Características:
- Alimentador de altíssima velocidade via Memory-Mapped uint16 (Zero gargalo de disco)
- Aceleração por GPU com TF32 e BFloat16 / Float16 com GradScaler
- Acúmulo de Gradientes (Gradient Accumulation) para batch size efetivo de 32k tokens
- Checkpoints automáticos periódicos e salvamento de estado do otimizador para AUTO-RESUME
- Amostragem gerativa periódica para inspecionar a emergência do raciocínio em tempo real
"""

import os
import sys
import time
import math
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suporte UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

from tokenizers import Tokenizer
from modelo_synapse_1b import BrainMoE1B, Config1B

def main():
    print("=" * 70)
    print("🧠 SYNAPSE 1B (MoE ~1.2B) — PRÉ-TREINAMENTO MASSIVO CONTÍNUO")
    print("=" * 70)

    # 1. Detecção de Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU Detectada: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("⚠️ Rodando em CPU (Modo de desenvolvimento/teste)")

    # 2. Localização das Pastas no Google Drive (5 TB) ou Local
    COLAB_DRIVE = Path("/content/drive/MyDrive/Synapse-1B")
    if COLAB_DRIVE.exists():
        DRIVE_ROOT = COLAB_DRIVE
    else:
        DRIVE_ROOT = Path("dados_synapse_1b")

    PASTA_DADOS = DRIVE_ROOT / "dados"
    PASTA_CHECKPOINTS = DRIVE_ROOT / "checkpoints"
    PASTA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    ARQUIVO_BIN = PASTA_DADOS / "treino_1b.bin"
    ARQUIVO_META = PASTA_DADOS / "meta_dados.json"
    CKPT_LATEST = PASTA_CHECKPOINTS / "synapse-1b-latest.pt"

    if not ARQUIVO_BIN.exists():
        print(f"❌ Erro: Arquivo de dados binários não encontrado em: {ARQUIVO_BIN}")
        print("   Execute primeiro: python preparar_dados_1b.py")
        sys.exit(1)

    # 3. Leitura do Memmap
    print(f"📁 Carregando base binária via Memory Mapping...")
    data = np.memmap(ARQUIVO_BIN, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    tamanho_mb = (total_tokens * 2) / (1024 * 1024)
    print(f"✅ Base Carregada: {total_tokens:,} tokens disponíveis ({tamanho_mb:.1f} MB em uint16)")

    # 4. Tokenizador
    tok_candidates = [
        Path("tokenizer.json"),
        Path("/content/cerebro-ia-102m/tokenizer.json"),
        Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m\tokenizer.json"),
    ]
    tok_file = None
    for cand in tok_candidates:
        if cand.exists():
            tok_file = cand
            break
    if not tok_file:
        cand_list = list(Path(".").rglob("tokenizer.json"))
        if cand_list:
            tok_file = cand_list[0]

    tok = Tokenizer.from_file(str(tok_file))
    pad_id = tok.token_to_id("<pad>") or 0
    eos_id = tok.token_to_id("<eos>") or 2

    # 5. Configuração e Inicialização do Modelo
    cfg = Config1B()
    model = BrainMoE1B(cfg).to(device)
    total_p, active_p = model.count_parameters()
    print(f"📊 Arquitetura Inicializada: {total_p:,} params totais | {active_p:,} ativos/token")

    # 6. Hiperparâmetros de Treinamento
    seq_len = cfg.context
    batch_size = 4 if torch.cuda.is_available() and vram_gb >= 16 else 2
    grad_accum_steps = 8  # batch efetivo: 32 seqs = 32.768 tokens por step
    tokens_per_step = batch_size * grad_accum_steps * seq_len
    
    total_steps = 100_000
    save_interval = 1000
    sample_interval = 250
    base_lr = 3e-4
    min_lr = 3e-5
    warmup_steps = 1000

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=base_lr, 
        betas=(0.9, 0.95), 
        eps=1e-8, 
        weight_decay=0.1
    )

    def get_lr(step):
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (base_lr - min_lr)

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and use_amp))

    # 7. Auto-Resume a partir do último Checkpoint no Drive
    start_step = 0
    if CKPT_LATEST.exists():
        print(f"\n🔄 Detectado checkpoint anterior em: {CKPT_LATEST}")
        try:
            ckpt = torch.load(CKPT_LATEST, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", 0)
            print(f"✅ Auto-Resume com sucesso! Retomando a partir do passo {start_step:,}...")
        except Exception as e:
            print(f"⚠️ Erro ao carregar checkpoint, iniciando do zero: {e}")

    # 8. Função de Obtenção de Lote Rápido em Memória
    def get_batch():
        max_idx = total_tokens - seq_len - 1
        indices = np.random.randint(0, max_idx, size=(batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in indices])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in indices])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    # 9. Função de Amostragem Textual em Tempo Real
    def gerar_amostra(prompt_texto):
        model.eval()
        p_ids = tok.encode(prompt_texto).ids
        input_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
        generated = list(p_ids)
        with torch.no_grad():
            for _ in range(60):
                curr = torch.tensor([generated[-cfg.context:]], device=device)
                logits, _ = model(curr)
                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits / 0.7, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
                if nxt == eos_id:
                    break
                generated.append(nxt)
        model.train()
        return tok.decode(generated)

    print("\n" + "=" * 70)
    print(f"⚡ DISPARANDO TREINAMENTO MASSIVO | Tokens/Step: {tokens_per_step:,}")
    print(f"💾 Checkpoints automáticos a cada {save_interval} steps em: {PASTA_CHECKPOINTS}")
    print("=" * 70)

    model.train()
    t_inicio_loop = time.time()
    
    for step in range(start_step, total_steps):
        t0 = time.time()
        
        # Ajuste de taxa de aprendizado com Warmup + Cosine
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        loss_acumulada = 0.0
        aux_loss_acumulada = 0.0

        for _ in range(grad_accum_steps):
            x, y = get_batch()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits, loss, aux_loss = model(x, targets=y)
                total_loss_batch = (loss + 0.01 * aux_loss) / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(total_loss_batch).backward()
            else:
                total_loss_batch.backward()

            loss_acumulada += loss.item() / grad_accum_steps
            aux_loss_acumulada += aux_loss.item() / grad_accum_steps

        # Atualização de Pesos com Gradiente Clip
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        dt = time.time() - t0
        tps = tokens_per_step / max(1e-5, dt)

        # Log a cada 10 passos
        if step % 10 == 0 or step == start_step:
            print(f"Passo [{step:>6}/{total_steps}] | Loss: {loss_acumulada:.4f} | Aux MoE: {aux_loss_acumulada:.4f} | LR: {lr:.2e} | {tps:,.0f} tok/s | dt: {dt:.2f}s")

        # Amostragem Gerativa
        if step > 0 and step % sample_interval == 0:
            print("\n" + "-" * 70)
            print(f"🔍 INSPEÇÃO COGNITIVA NO PASSO {step}:")
            amostra = gerar_amostra("A inteligência artificial é")
            print(f"Prompt: 'A inteligência artificial é...'\n{amostra}")
            print("-" * 70 + "\n")

        # Salvamento de Checkpoint no Drive (5 TB)
        if step > 0 and step % save_interval == 0:
            ckpt_path_step = PASTA_CHECKPOINTS / f"synapse-1b-step-{step}.pt"
            print(f"\n💾 Salvando Checkpoint em: {ckpt_path_step}...")
            estado_salvar = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "loss": loss_acumulada,
                "type": "Synapse-1B-SparseMoE"
            }
            torch.save(estado_salvar, ckpt_path_step)
            torch.save(estado_salvar, CKPT_LATEST)
            print("✅ Checkpoint e Latest salvos com sucesso no Google Drive!\n")

    print("\n" + "=" * 70)
    print("🎯 PRÉ-TREINAMENTO MASSIVO DA SYNAPSE 1B CONCLUÍDO!")
    print("=" * 70)

if __name__ == "__main__":
    main()
