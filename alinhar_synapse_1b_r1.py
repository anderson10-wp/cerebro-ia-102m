#!/usr/bin/env python3
"""
ALINHAMENTO COGNITIVO SYNAPSE-R1 (SFT COM MODO PENSAMENTO PROFUNDO)
Desenvolvido por Nandinho.
Aplica a destilação de raciocínio com auto-verificação (<think>)
sobre o checkpoint pré-treinado de 1.2B da Synapse.
Entrada: /content/drive/MyDrive/Synapse-1B/checkpoints/synapse-1b-latest.pt
Dataset: dataset_synapse_r1_reasoning.jsonl (com máscara -100)
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
    print("=" * 70)
    print("🧠 SYNAPSE-R1 — ALINHAMENTO COGNITIVO COM MODO PENSAMENTO PROFUNDO")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU Ativa: {gpu_name} ({vram:.1f} GB VRAM)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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
    ]
    ckpt_base = None
    for cp in ckpt_candidates:
        if cp.exists():
            ckpt_base = cp
            break

    if not ckpt_base:
        print(f"❌ Nenhum checkpoint de pré-treino encontrado em: {PASTA_CHECKPOINTS}")
        print("   Aguarde o pré-treinamento salvar o primeiro checkpoint.")
        sys.exit(1)

    print(f"📂 Checkpoint Base Selecionado: {ckpt_base}")

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
    
    # Limpa possíveis prefixos
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state[k[len("_orig_mod."):]] = v
        else:
            clean_state[k] = v
            
    model.load_state_dict(clean_state)
    model.to(device)
    print("✅ Cérebro Synapse 1.2B carregado com sucesso!")

    # 4. Dataset R1 e DataLoader
    print(f"📋 Carregando Dataset R1 ({DATASET_FILE})...")
    ds = MaskedReasoningDataset(DATASET_FILE, tok, max_len=1024)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    print(f"✅ {len(ds)} amostras carregadas para alinhamento cognitivo!")

    # 5. Otimizador e Scheduler SFT
    epochs = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=1e-5)

    print("\n" + "=" * 70)
    print(f"⚡ INICIANDO CALIBRAÇÃO R1 ({epochs} Épocas | LR: 3.5e-5 -> 1e-5)...")
    print("=" * 70)

    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        steps = 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits, loss, aux_loss = model(input_ids[:, :-1], targets=labels[:, 1:])
            total_batch_loss = loss + 0.01 * aux_loss
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            steps += 1
            
        dt = time.time() - t0
        avg_loss = total_loss / steps
        cur_lr = scheduler.get_last_lr()[0]
        print(f"Época [{ep}/{epochs}] | Loss nas Respostas: {avg_loss:.4f} | LR: {cur_lr:.2e} | Tempo: {dt:.2f}s")

    # 6. Salvar Modelo Final
    print(f"\n💾 Salvando Modelo Final com Modo Pensamento em: {MODEL_OUTPUT}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_state": model.state_dict(),
        "config": cfg,
        "type": "Synapse-1B-R1-Reasoning-MoE"
    }, MODEL_OUTPUT)
    print("✅ Checkpoint Final R1 salvo com sucesso no Google Drive!")

    # 7. Testes de Verificação do Modo Pensamento
    print("\n" + "=" * 70)
    print("📋 BATERIA DE HOMOLOGAÇÃO: MODO PENSAMENTO PROFUNDO (R1)")
    print("=" * 70)
    model.eval()

    def responder(pergunta):
        prompt = f"### Usuário:\n{pergunta}\n\n### Assistente:\n"
        input_ids = torch.tensor([tok.encode(prompt).ids], device=device)
        generated = input_ids[0].tolist()
        
        with torch.no_grad():
            for _ in range(350):
                curr = torch.tensor([generated[-cfg.context:]], device=device)
                logits, _ = model(curr)
                next_token_logits = logits[0, -1, :].clone()
                next_token_logits[pad_token_id] = -float('inf')
                
                # Amostragem Top-K (40) com temp 0.6
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
        "Escreva uma função em Python que soma dois números",
        "Qual a diferença entre uma lista e uma tupla em Python?",
        "Explique em duas frases o que é a gravidade.",
        "Quem é você e qual é o seu objetivo em uma frase simples?",
        "como eu faço novas amizades?"
    ]

    for q in testes:
        print(f"\nVocê: {q}")
        resp = responder(q)
        print(f"🤖 Synapse-R1:\n{resp}\n{'-'*50}")

    # Chat Interativo Livre
    print("\n" + "=" * 70)
    print("💬 SYNAPSE-R1 ONLINE — CHAT LIVRE (Digite 'sair' para encerrar)")
    print("=" * 70)
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
