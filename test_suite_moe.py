import os, sys, math, time, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("🧠 BATERIA DE AVALIAÇÃO DO MODO PENSAR — CÉREBRO MoE 500M")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de teste: {device}")

# Importa a arquitetura do train_moe_500m_colab
sys.path.append(os.path.abspath("."))
from train_moe_500m_colab import Config, BrainMoE

cfg = Config()
model = BrainMoE(cfg)

# Carrega o checkpoint treinado
checkpoint_candidates = [
    "cerebro-moe-500m-modo-pensar.pt",
    "/content/cerebro-ia-102m/cerebro-moe-500m-modo-pensar.pt",
    "/content/drive/MyDrive/cerebro-moe-500m-modo-pensar.pt"
]

loaded = False
for ckpt_path in checkpoint_candidates:
    if os.path.exists(ckpt_path):
        print(f"📦 Carregando modelo treinado de: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location='cpu')
        state = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
        model.load_state_dict(state)
        loaded = True
        break

if not loaded:
    print("❌ Checkpoint não encontrado nos caminhos padrão!")
    sys.exit(1)

model = model.to(device)
model.eval()
print("✅ Cérebro MoE 500M pronto para os testes!\n")

# Tokenizador
from tokenizers import Tokenizer
tok_candidates = list(Path('.').rglob('tokenizer.json'))
tok = Tokenizer.from_file(str(tok_candidates[0]))

perguntas = [
    "Se 3 pessoas constroem 3 muros em 3 dias, quantos dias 9 pessoas levam para construir 9 muros?",
    "Em uma corrida, se você ultrapassa o segundo colocado, em que posição você fica?",
    "O que é mais pesado: 1 quilo de chumbo ou 1 quilo de algodão?",
    "Por que não sentimos a Terra girando a mais de 1600 km/h?",
    "Se 5 máquinas levam 5 minutos para fazer 5 peças, quanto tempo 100 máquinas levam para fazer 100 peças?",
    "Quem é você e qual é a sua missão?"
]

def resolver(pergunta):
    print("=" * 70)
    print(f"👉 PERGUNTA: {pergunta}")
    print("=" * 70)
    prompt = f"Pergunta: {pergunta}\nResposta: <think>\n"
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(260):
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(ids)
            else:
                logits, _ = model(ids)
            probs = F.softmax(logits[0, -1, :] / 0.65, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
            if ids.shape[1] >= cfg.context:
                break
                
    saida = tok.decode(ids[0].tolist())
    print(saida)
    print("\n")

for p in perguntas:
    resolver(p)
