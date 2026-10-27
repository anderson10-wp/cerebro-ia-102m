#!/usr/bin/env python3
"""
ALINHAMENTO COGNITIVO SYNAPSE-R1 (SFT COM MODO PENSAMENTO PROFUNDO)
Desenvolvido por Nandinho.
Aplica a destilação de raciocínio com auto-verificação (<think>)
sobre o checkpoint pré-treinado de 1.2B da Synapse.
Entrada: /content/drive/MyDrive/Synapse-1B/checkpoints/synapse-1b-latest.pt
Dataset: dataset_synapse_r1_reasoning.jsonl (com máscara -100 no prompt)
Saída: /content/drive/MyDrive/Synapse-1B/synapse-1b-r1-final.pt
"""

import os
import sys
import time
import math
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Suporte UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

from tokenizers import Tokenizer
from modelo_synapse_1b import BrainMoE1B, Config1B

# -----------------------------------------------------------------------------
# DATASET SFT COM MÁSCARA -100 (Loss apenas no <think> e na resposta)
# -----------------------------------------------------------------------------
class MaskedReasoningDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        pad_id = tokenizer.token_to_id("<pad>") or 0
        eos_id = tokenizer.token_to_id("<eos>") or 2

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                p = d["pergunta"].strip()
                t = d["pensamento"].strip()
                r = d["resposta"].strip()
                
                prompt = f"### Usuário:\n{p}\n\n### Assistente:\n"
                full_text = f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}"
                
                prompt_ids = tokenizer.encode(prompt).ids
                full_ids = tokenizer.encode(full_text).ids + [eos_id]
                
                if len(full_ids) > max_len:
                    full_ids = full_ids[:max_len]
                
                # Máscara -100 no prompt do usuário (o modelo não é penalizado pelo que o usuário perguntou)
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
    print("🧠 SYNAPSE-R1 — ALINHAMENTO COGNITIVO COM MODO PENSAMENTO PROFUNDO")
    print("   Arquitetura: 1.21B Parâmetros (Sparse MoE 12x Top-2)")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False
    amp_dtype = torch.float32

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU Ativa: {gpu_name} ({vram:.1f} GB VRAM)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_amp = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"⚡ Aceleração AMP Ativa: {amp_dtype}")
    else:
        print("⚠️ Rodando em CPU!")

    # 1. Caminhos no Drive de 5 TB
    COLAB_DRIVE = Path("/content/drive/MyDrive/Synapse-1B")
    if COLAB_DRIVE.exists():
        DRIVE_ROOT = COLAB_DRIVE
    else:
        DRIVE_ROOT = Path("dados_synapse_1b")

    PASTA_CHECKPOINTS = DRIVE_ROOT / "checkpoints"
    MODEL_OUTPUT = DRIVE_ROOT / "synapse-1b-r1-final.pt"
    DATASET_FILE = Path("dataset_synapse_r1_reasoning.jsonl")

    # Localizar Checkpoint Pré-Treinado Base
    ckpt_candidates = [
        PASTA_CHECKPOINTS / "synapse-1b-latest.pt",
        PASTA_CHECKPOINTS / "synapse-1b-step-2000.pt",
        PASTA_CHECKPOINTS / "synapse-1b-step-1000.pt",
        Path("synapse-1b-latest.pt")
    ]
    ckpt_base = None
    for cp in ckpt_candidates:
        if cp.exists():
            ckpt_base = cp
            break

    if not ckpt_base:
        print(f"❌ Nenhum checkpoint de pré-treino encontrado em: {PASTA_CHECKPOINTS}")
        print("   Certifique-se de que o pré-treinamento gerou o checkpoint synapse-1b-latest.pt.")
        sys.exit(1)

    print(f"📂 Checkpoint Base Pré-Treinado: {ckpt_base}")

    # 2. Tokenizador
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
    tok = Tokenizer.from_file(str(tok_file))
    pad_token_id = tok.token_to_id("<pad>") or 0
    eos_token_id = tok.token_to_id("<eos>") or 2

    # 3. Carregar Modelo 1.2B
    cfg = Config1B()
    model = BrainMoE1B(cfg)
    
    print(f"📥 Carregando pesos pré-treinados...")
    raw = torch.load(ckpt_base, map_location="cpu")
    state_dict = raw.get("model_state_dict", raw.get("model_state", raw))
    
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state[k[len("_orig_mod."):]] = v
        else:
            clean_state[k] = v
            
    model.load_state_dict(clean_state)
    model.to(device)
    total_p, active_p = model.count_parameters()
    print(f"✅ Synapse 1.2B carregado! Total: {total_p/1e6:.1f}M | Ativos: {active_p/1e6:.1f}M por token.")

    # 4. Dataset R1 e DataLoader
    print(f"📋 Compilando Dataset R1 ({DATASET_FILE.name})...")
    ds = MaskedReasoningDataset(DATASET_FILE, tok, max_len=1024)
    # Batch size 4 em A100 roda liso com ~6GB VRAM
    batch_size = 4 if torch.cuda.is_available() else 1
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"✅ {len(ds):,} amostras de raciocínio profundo prontas para alinhamento!")

    # 5. Otimizador e Scheduler SFT
    epochs = 3
    total_steps = epochs * len(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=8.0e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    print("\n" + "=" * 72)
    print(f"⚡ DISPARANDO ALINHAMENTO COGNITIVO R1 ({epochs} Épocas | {total_steps} Passos)")
    print(f"🎯 Meta: Redução da Loss nas respostas e estabilização de tags <think>")
    print("=" * 72)

    model.train()
    global_step = 0
    t_inicio_sft = time.time()

    for ep in range(1, epochs + 1):
        t0_ep = time.time()
        total_loss = 0.0
        steps_ep = 0

        for batch_idx, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits, loss, aux_loss = model(input_ids[:, :-1], targets=labels[:, 1:])
                total_batch_loss = loss + 0.01 * aux_loss

            if scaler.is_enabled():
                scaler.scale(total_batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            steps_ep += 1
            global_step += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Época [{ep}/{epochs}] | Batch [{batch_idx+1}/{len(loader)}] | Loss Resposta: {loss.item():.4f} | Aux MoE: {aux_loss.item():.3f} | LR: {cur_lr:.2e}")

        avg_loss = total_loss / max(1, steps_ep)
        dt_ep = time.time() - t0_ep
        print(f"--> [FIM DA ÉPOCA {ep}] Loss Média: {avg_loss:.4f} | Tempo: {dt_ep:.1f}s ({dt_ep/60:.2f} min)")

    tempo_total = (time.time() - t_inicio_sft) / 60
    print(f"\n✨ ALINHAMENTO COGNITIVO CONCLUÍDO EM {tempo_total:.2f} MINUTOS!")

    # 6. Salvar Modelo Final
    print(f"\n💾 Gravando Checkpoint Definitivo Synapse-1B-R1 em: {MODEL_OUTPUT}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_state": model.state_dict(),
        "config": cfg,
        "type": "Synapse-1B-R1-Reasoning-MoE"
    }, MODEL_OUTPUT)
    print("✅ Checkpoint Final R1 salvo com segurança no Google Drive!")

    # 7. Testes de Verificação do Modo Pensamento
    print("\n" + "=" * 72)
    print("📋 BATERIA DE HOMOLOGAÇÃO: MODO PENSAMENTO PROFUNDO (R1)")
    print("=" * 72)
    model.eval()

    def responder(pergunta):
        prompt = f"### Usuário:\n{pergunta}\n\n### Assistente:\n"
        input_ids = torch.tensor([tok.encode(prompt).ids], device=device)
        generated = input_ids[0].tolist()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                for _ in range(350):
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
        "Qual a diferença entre uma lista e uma tupla em Python?",
        "Explique em duas frases o que é a gravidade.",
        "Quem é você e quem te criou?"
    ]

    for q in testes:
        print(f"\nVocê: {q}")
        resp = responder(q)
        print(f"🤖 Synapse-R1:\n{resp}\n{'-'*50}")

    # Chat Interativo Livre no Colab
    print("\n" + "=" * 72)
    print("💬 SYNAPSE-R1 ONLINE — CHAT INTERATIVO NO COLAB")
    print("=" * 72)
    while True:
        try:
            q = input("\nVocê: ")
            if q.strip().lower() in ["sair", "exit", "quit", "q"]:
                break
            if not q.strip(): continue
            resp = responder(q)
            print(f"\n🤖 Synapse-R1:\n{resp}\n{'-'*50}")
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main()
