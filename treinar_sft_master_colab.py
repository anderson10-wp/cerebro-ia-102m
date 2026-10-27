#!/usr/bin/env python3
"""
TREINAMENTO SFT MASTER DO SYNAPSE (COM MÁSCARA -100)
Carrega a base contínua do Drive (/content/drive/MyDrive/cerebro-moe-500m-continuo.pt)
e calibra em 1.668 amostras balanceadas de conversação, fatos, matemática e identidade.
"""

import os, sys, math, time, json
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

print("=" * 70)
print("🧠 SYNAPSE MoE 500M — CALIBRAÇÃO SFT MASTER (MÁSCARA -100)")
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

# Caminhos
DRIVE_BASE = Path("/content/drive/MyDrive")
MODEL_INPUT = DRIVE_BASE / "cerebro-moe-500m-continuo.pt"
MODEL_OUTPUT = DRIVE_BASE / "cerebro-moe-500m-synapse.pt"
DATASET_FILE = Path("dataset_sft_master.jsonl")
TOKENIZER_FILE = Path("tokenizer.json")

if not TOKENIZER_FILE.exists():
    # Procura no repo
    cand = list(Path(".").rglob("tokenizer.json"))
    if cand:
        TOKENIZER_FILE = cand[0]

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
pad_token_id = tokenizer.token_to_id("<pad>") or 0

# -----------------------------------------------------------------------------
# ARQUITETURA MoE (Idêntica ao pre-treino)
# -----------------------------------------------------------------------------
@dataclass
class Config:
    dim: int = 640
    n_layers: int = 16
    n_heads: int = 10
    vocab_size: int = 32768
    context: int = 512
    hidden_dim: int = 1792
    num_experts: int = 8
    num_experts_per_tok: int = 2

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.head_dim = cfg.dim // cfg.n_heads
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(out)

class MoESwiGLUExpert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.dim, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([MoESwiGLUExpert(cfg) for _ in range(cfg.num_experts)])
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        logits = self.router(x_flat)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.cfg.num_experts_per_tok, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        out_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                idx = mask.nonzero().squeeze(-1)
                tok_x = x_flat[idx]
                tok_w = weights[idx, (indices[idx] == i).nonzero(as_tuple=True)[1]].unsqueeze(-1)
                out_flat[idx] += tok_w * expert(tok_x)
        return out_flat.view(B, T, C), 0.0

class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.moe = MoELayer(cfg)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        moe_out, _ = self.moe(self.norm2(x))
        return x + moe_out

class BrainMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.context, cfg.dim)
        self.blocks = nn.ModuleList([MoEBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

# -----------------------------------------------------------------------------
# DATASET SFT COM MÁSCARA -100
# -----------------------------------------------------------------------------
class MaskedChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                tf = d.get("treino_formatado", "")
                if "### Usuário:\n" in tf and "\n\n### Assistente:\n" in tf:
                    parts = tf.split("\n\n### Assistente:\n")
                    prompt_str = parts[0] + "\n\n### Assistente:\n"
                    resp_str = parts[1]
                    
                    p_ids = tokenizer.encode(prompt_str).ids
                    r_ids = tokenizer.encode(resp_str).ids + [tokenizer.token_to_id("<eos>") or 2]
                    
                    if len(p_ids) + len(r_ids) > max_len:
                        r_ids = r_ids[:max_len - len(p_ids)]
                        
                    input_ids = p_ids + r_ids
                    # Mascara o prompt com -100
                    labels = [-100] * len(p_ids) + r_ids
                    
                    pad_len = max_len - len(input_ids)
                    input_ids += [pad_token_id] * pad_len
                    labels += [-100] * pad_len
                    
                    self.examples.append((
                        torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(labels, dtype=torch.long)
                    ))
        print(f"✅ {len(self.examples)} conversas carregadas para SFT!")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def main():
    cfg = Config()
    
    # Carregar modelo
    print(f"📦 Carregando modelo base: {MODEL_INPUT}...")
    if not MODEL_INPUT.exists():
        print(f"⚠️ {MODEL_INPUT} não encontrado! Usando modelo local...")
        local_cand = list(Path(".").rglob("cerebro-moe-500m-continuo.pt"))
        if local_cand:
            model_path = local_cand[0]
        else:
            raise FileNotFoundError("Modelo base não encontrado!")
    else:
        model_path = MODEL_INPUT

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    
    model = BrainMoE(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    print("✅ Cérebro MoE pronto!")

    # Dataset e DataLoader
    dataset = MaskedChatDataset(DATASET_FILE, tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    epochs = 4
    print("\n" + "=" * 70)
    print(f"⚡ INICIANDO CALIBRAÇÃO COM MÁSCARA -100 ({epochs} Épocas)...")
    print("=" * 70)

    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        steps = 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward
            logits = model(input_ids[:, :-1])
            loss = criterion(logits.reshape(-1, cfg.vocab_size), labels[:, 1:].reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        dt = time.time() - t0
        avg_loss = total_loss / steps
        print(f"Época [{ep}/{epochs}] | Loss nas Respostas: {avg_loss:.4f} | Tempo: {dt:.2f}s")

    # Salvar Checkpoint
    print(f"\n💾 Salvando checkpoint calibrado em: {MODEL_OUTPUT}...")
    torch.save({"model_state": model.state_dict(), "cfg": cfg}, MODEL_OUTPUT)
    print("✅ Checkpoint salvo com sucesso no Google Drive!")

    # Chat Interativo
    print("\n" + "=" * 70)
    print("💬 SYNAPSE ONLINE — CHAT INTERATIVO ATIVADO")
    print("=" * 70)
    model.eval()

    def responder(pergunta):
        prompt = f"### Usuário:\n{pergunta}\n\n### Assistente:\n"
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
        generated = input_ids[0].tolist()
        
        with torch.no_grad():
            for _ in range(160):
                curr = torch.tensor([generated[-512:]], device=device)
                logits = model(curr)[0, -1, :]
                logits[pad_token_id] = -float('inf')
                
                # Temperatura moderada
                probs = F.softmax(logits / 0.7, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
                if nxt == (tokenizer.token_to_id("<eos>") or 2):
                    break
                generated.append(nxt)
                
        resposta_txt = tokenizer.decode(generated[len(input_ids[0]):])
        return resposta_txt.strip()

    testes = [
        "oi",
        "quem é você?",
        "preciso de sua ajuda",
        "quanto é 29 + 20?",
        "vale a pena aprender Python hoje em dia?"
    ]
    for q in testes:
        print(f"\nVocê: {q}")
        resp = responder(q)
        print(f"🤖 Synapse:\n{resp}\n{'-'*50}")

if __name__ == "__main__":
    main()
