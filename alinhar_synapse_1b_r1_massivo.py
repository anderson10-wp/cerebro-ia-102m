#!/usr/bin/env python3
"""
ALINHAMENTO COGNITIVO MASSIVO SYNAPSE-R1 V2 (22.000+ AMOSTRAS)
Projeto Synapse — Desenvolvido por Nandinho.

Executa o treinamento supervisionado de raciocínio profundo estilo DeepSeek-R1
com mascaramento de loss estrito (-100) na GPU NVIDIA A100 (Google Colab).
"""

import os
import sys
import glob
import time
import math
import json
from pathlib import Path

# Suporte UTF-8 no Windows e Linux
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from modelo_synapse_1b import BrainMoE1B, Config1B

# Registra classes seguras para PyTorch 2.6+
try:
    torch.serialization.add_safe_globals([Config1B])
except Exception:
    pass

# -----------------------------------------------------------------------------
# DATASET SFT COM MÁSCARA -100 (Loss apenas no <think> e na resposta)
# -----------------------------------------------------------------------------
class MaskedReasoningDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        pad_id = tokenizer.token_to_id("<|pad|>") or 0
        eos_id = tokenizer.token_to_id("<|eos|>") or 2

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                p = d.get("pergunta", "").strip()
                t = d.get("pensamento", "").strip()
                r = d.get("resposta", "").strip()
                
                if t:
                    prompt = f"### Usuário:\n{p}\n\n### Assistente:\n"
                    full_text = f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}"
                else:
                    prompt = f"### Usuário:\n{p}\n\n### Assistente:\n"
                    full_text = f"### Usuário:\n{p}\n\n### Assistente:\n{r}"
                
                prompt_ids = tokenizer.encode(prompt).ids
                full_ids = tokenizer.encode(full_text).ids + [eos_id]
                
                if len(full_ids) > max_len:
                    full_ids = full_ids[:max_len]
                
                # Máscara -100 no prompt do usuário
                labels = list(full_ids)
                prompt_len = min(len(prompt_ids), len(labels))
                for i in range(prompt_len):
                    labels[i] = -100
                    
                # Padding até max_len
                pad_len = max_len - len(full_ids)
                if pad_len > 0:
                    input_ids = full_ids + [pad_id] * pad_len
                    labels = labels + [-100] * pad_len
                else:
                    input_ids = full_ids[:max_len]
                    labels = labels[:max_len]
                    
                self.samples.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def main():
    print("=" * 72)
    print("🧠 SYNAPSE-R1 V2 — ALINHAMENTO COGNITIVO MASSIVO (22k AMOSTRAS)")
    print("   Arquitetura: 1.21B Parâmetros (Sparse MoE 12x Top-2)")
    print("   Projeto Soberano — Desenvolvido por Nandinho")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False
    amp_dtype = torch.float32

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU Ativa: {gpu_name} ({vram:.1f} GB VRAM)")
        
        # A100 e GPUs modernas suportam bfloat16
        if torch.cuda.is_bf16_supported():
            use_amp = True
            amp_dtype = torch.bfloat16
            print("⚡ Aceleração AMP Ativa: torch.bfloat16 (Alta precisão numérica)")
        else:
            use_amp = True
            amp_dtype = torch.float16
            print("⚡ Aceleração AMP Ativa: torch.float16")
    else:
        print("💻 Executando em CPU.")

    # 1. Carrega Tokenizador BPE
    tok_candidates = [
        Path("/content/cerebro-ia-102m/tokenizer.json"),
        Path("tokenizer.json"),
        Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m\tokenizer.json")
    ]
    tok_path = next((p for p in tok_candidates if p.exists()), None)
    if not tok_path:
        print("❌ Tokenizador tokenizer.json não encontrado!")
        sys.exit(1)
        
    tok = Tokenizer.from_file(str(tok_path))
    eos_token_id = tok.token_to_id("<|eos|>") or 2
    pad_token_id = tok.token_to_id("<|pad|>") or 0

    # 2. Localização do Checkpoint Base (prioriza modelo limpo de 4.8 GB)
    print("\n🔍 Buscando checkpoint base para treinamento...")
    ckpt_candidates = [
        "/content/drive/MyDrive/Synapse-1B/checkpoints/synapse-1b-r1-final.pt",
        "/content/drive/MyDrive/checkpoints/synapse-1b-r1-final.pt",
        "synapse-1b-r1-final.pt",
        "/content/drive/MyDrive/Synapse-1B/checkpoints/synapse-1b-step-10000.pt",
        "/content/drive/MyDrive/dados_synapse_1b/checkpoints/synapse-1b-step-10000.pt",
        "synapse-1b-step-10000.pt",
        "synapse-1b-latest.pt",
        str(Path.home() / "Downloads" / "synapse-1b-r1-final.pt"),
        str(Path.home() / "Downloads" / "synapse-1b-step-10000.pt")
    ]
    
    ckpt_base = next((c for c in ckpt_candidates if Path(c).exists()), None)
    
    # Se não achou na lista direta, faz busca recursiva no Drive
    if not ckpt_base and Path("/content/drive/MyDrive").exists():
        found_r1 = glob.glob("/content/drive/MyDrive/**/synapse-1b-r1-final.pt", recursive=True)
        if found_r1:
            ckpt_base = found_r1[0]
        else:
            found_10k = glob.glob("/content/drive/MyDrive/**/synapse-1b-step-10000.pt", recursive=True)
            if found_10k:
                ckpt_base = found_10k[0]

    if not ckpt_base:
        print("❌ Nenhum checkpoint de 1B encontrado! Conecte o Google Drive com o arquivo .pt.")
        sys.exit(1)

    print(f"📂 Checkpoint Base Localizado: {ckpt_base} ({Path(ckpt_base).stat().st_size / (1024**3):.2f} GB)")

    # 3. Instanciação e Carregamento do Modelo MoE
    cfg = Config1B(
        vocab_size=tok.get_vocab_size(),
        dim=768,
        layers=20,
        n_heads=12,
        head_dim=64,
        hidden_dim=2048,
        num_experts=12,
        top_k=2,
        context=1024
    )
    model = BrainMoE1B(cfg).to(device)

    print("📥 Carregando pesos pré-treinados com mapeamento de memória seguro (mmap)...")
    try:
        raw = torch.load(ckpt_base, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        raw = torch.load(ckpt_base, map_location="cpu", weights_only=False)
        
    state = raw.get("model_state_dict", raw.get("model_state", raw))
    clean_state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    
    # Libera imediatamente a RAM do sistema operacional
    del raw, state, clean_state
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    active_params = (cfg.dim * cfg.vocab_size * 2 + cfg.layers * (4 * cfg.dim**2 + cfg.top_k * 3 * cfg.dim * cfg.hidden_dim)) / 1e6
    print(f"✅ Synapse 1.2B carregado! Total: {total_params:.1f}M | Ativos: {active_params:.1f}M por token.")

    # 4. Carrega o Dataset Massivo 22k
    dataset_candidates = [
        Path("/content/cerebro-ia-102m/dataset_synapse_r1_20k.jsonl"),
        Path("dataset_synapse_r1_20k.jsonl"),
        Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m\dataset_synapse_r1_20k.jsonl")
    ]
    dataset_path = next((d for d in dataset_candidates if d.exists()), None)
    if not dataset_path:
        print("❌ Dataset dataset_synapse_r1_20k.jsonl não encontrado!")
        sys.exit(1)

    print(f"\n📋 Compilando Dataset Massivo R1 ({dataset_path.name})...")
    ds = MaskedReasoningDataset(str(dataset_path), tok, max_len=cfg.context)
    print(f"✅ {len(ds):,} amostras de raciocínio profundo prontas para alinhamento!")

    # 5. Hiperparâmetros de Treinamento
    # Batch size 4 com grad accum 8 = batch efetivo de 32 sequências
    batch_size = 4 if torch.cuda.is_available() else 1
    grad_accum_steps = 8 if torch.cuda.is_available() else 2
    epochs = 3
    lr = 3e-5
    min_lr = 3e-6

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    total_steps = (len(loader) // grad_accum_steps) * epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    
    # Scheduler Cosine com Warmup de 3%
    warmup_steps = int(total_steps * 0.03)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    print("=" * 72)
    print(f"⚡ DISPARANDO ALINHAMENTO COGNITIVO R1 MASSIVO ({epochs} Épocas | {total_steps:,} Passos)")
    print(f"🎯 Batch Efetivo: {batch_size * grad_accum_steps} | Sequências por Época: {len(ds):,}")
    print("=" * 72)

    model.train()
    global_step = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_aux = 0.0
        
        for step, (x, y) in enumerate(loader, 1):
            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                logits, aux_loss = model(x)
                shift_logits = logits[..., :-1, :].contiguous().view(-1, cfg.vocab_size)
                shift_labels = y[..., 1:].contiguous().view(-1)
                
                loss_ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                loss_total = (loss_ce + 0.01 * aux_loss) / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_total).backward()
            else:
                loss_total.backward()

            accum_loss += loss_ce.item()
            accum_aux += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else float(aux_loss)

            if step % grad_accum_steps == 0 or step == len(loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 25 == 0 or global_step == 1:
                    avg_ce = accum_loss / grad_accum_steps
                    avg_aux = accum_aux / grad_accum_steps
                    cur_lr = optimizer.param_groups[0]['lr']
                    elapsed = time.time() - t_start
                    tok_sec = (global_step * batch_size * grad_accum_steps * cfg.context) / max(elapsed, 1e-5)
                    print(f"Época [{epoch}/{epochs}] | Passo [{global_step:,}/{total_steps:,}] | "
                          f"Loss Resp: {avg_ce:.4f} | Aux MoE: {avg_aux:.3f} | LR: {cur_lr:.2e} | {tok_sec:.0f} tok/s", flush=True)
                    accum_loss = 0.0
                    accum_aux = 0.0

        # Salvamento de Checkpoint ao final de cada época
        save_dir = Path("/content/drive/MyDrive/Synapse-1B/checkpoints")
        if not save_dir.exists():
            save_dir = Path("./checkpoints")
            save_dir.mkdir(parents=True, exist_ok=True)
            
        epoch_path = save_dir / f"synapse-1b-r1-v2-epoch-{epoch}.pt"
        print(f"\n💾 Salvando checkpoint da Época {epoch} em: {epoch_path}...", flush=True)
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, str(epoch_path))
        print("✅ Checkpoint da época salvo com sucesso!")

    # 6. Salvamento Final
    final_path = save_dir / "synapse-1b-r1-v2-final.pt"
    print(f"\n💾 Salvando Modelo Final Synapse-R1 V2 em: {final_path}...", flush=True)
    torch.save({
        "config": cfg,
        "model_state_dict": model.state_dict()
    }, str(final_path))
    print(f"✨ ALINHAMENTO COGNITIVO V2 CONCLUÍDO COM SUCESSO EM {time.time()-t_start:.1f}s!")

    # 7. Bateria de Testes Pós-Treino
    print("\n" + "=" * 72)
    print("🧪 BATERIA DE AVALIAÇÃO COGNITIVA PÓS-TREINAMENTO V2")
    print("=" * 72)
    model.eval()

    def responder(pergunta):
        prompt = f"### Usuário:\n{pergunta}\n\n### Assistente:\n"
        input_ids = torch.tensor([tok.encode(prompt).ids], device=device)
        generated = input_ids[0].tolist()
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                for _ in range(400):
                    curr = torch.tensor([generated[-cfg.context:]], device=device)
                    logits, _ = model(curr)
                    next_token_logits = logits[0, -1, :].clone()
                    next_token_logits[pad_token_id] = -float('inf')
                    
                    top_k = 40
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[-1]] = -float('inf')
                    probs = F.softmax(next_token_logits / 0.6, dim=-1)
                    nxt = torch.multinomial(probs, 1).item()
                    if nxt == eos_token_id:
                        break
                    generated.append(nxt)
                    
        return tok.decode(generated[len(input_ids[0]):]).strip()

    testes = [
        "Me dê uma lista de 3 frutas amarelas.",
        "Se eu tenho 3 maçãs e como 1, depois ganho mais 5, com quantas maçãs eu fico? Mostre apenas a conta.",
        "Quanto é 29 + 20?",
        "Como fazer um loop for em Python?",
        "Qual a diferença entre dolo eventual e culpa consciente no direito penal?",
        "Explique em duas frases o que é a gravidade.",
        "Quem é você e quem te criou?"
    ]

    for q in testes:
        print(f"\nVocê: {q}")
        resp = responder(q)
        print(f"🤖 Synapse-R1 V2:\n{resp}\n{'-'*50}")

    print("\n" + "=" * 72)
    print("🚀 TREINAMENTO E VALIDAÇÃO CONCLUÍDOS!")
    print("=" * 72)

if __name__ == "__main__":
    main()
