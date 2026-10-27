#!/usr/bin/env python3
"""
TREINAMENTO SFT MASTER DO SYNAPSE (COM MÁSCARA -100)
Carrega a base contínua do Drive (/content/drive/MyDrive/cerebro-moe-500m-continuo.pt)
e calibra em 1.668 amostras balanceadas de conversação, fatos, matemática e identidade.
Arquitetura 100% compatível com RoPE, QKV fundido e Experts (gate, up, down).
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
DATASET_FILE = Path("dataset_sft_master_v3.jsonl")
if not DATASET_FILE.exists():
    DATASET_FILE = Path("dataset_sft_master_v2.jsonl")
TOKENIZER_FILE = Path("tokenizer.json")

if not TOKENIZER_FILE.exists():
    cand = list(Path(".").rglob("tokenizer.json"))
    if cand:
        TOKENIZER_FILE = cand[0]

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
pad_token_id = tokenizer.token_to_id("<pad>") or 0

# -----------------------------------------------------------------------------
# ARQUITETURA MoE 500M OFICIAL (ROPE + QKV FUNDIDO + EXPERTS GATE/UP/DOWN)
# -----------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 32768
    dim: int = 640
    layers: int = 16
    heads: int = 10
    hidden_dim: int = 1792
    context: int = 1024
    rope_base: float = 10000.0
    num_experts: int = 8
    top_k: int = 2

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        inv = cfg.rope_base ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        angles = torch.outer(torch.arange(cfg.context).float(), inv)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def rotate(self, x):
        length = x.size(-2)
        cos = self.cos[:length].to(x.dtype)[None, None]
        sin = self.sin[:length].to(x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        z = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v, is_causal=True)
        return self.out(z.transpose(1, 2).contiguous().view(b, t, d))

class MoESwiGLUExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.dim, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            MoESwiGLUExpert(cfg.dim, cfg.hidden_dim) for _ in range(cfg.num_experts)
        ])

    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i)
            if mask.any():
                token_indices, k_pos = torch.where(mask)
                tokens_in = x_flat[token_indices]
                expert_out = expert(tokens_in)
                weights = top_k_weights[token_indices, k_pos].unsqueeze(-1)
                out.index_add_(0, token_indices, (expert_out * weights).to(out.dtype))
        
        # Load balancing loss
        f = (top_k_indices.unsqueeze(-1) == torch.arange(self.num_experts, device=x.device)).float().sum(dim=[0, 1]) / (b * t * self.top_k)
        P = probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f * P)
        return out.view(b, t, d), aux_loss

class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.moe = MoELayer(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        moe_out, aux_loss = self.moe(self.norm2(x))
        return x + moe_out, aux_loss

class BrainMoE(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
        self.blocks = nn.ModuleList([MoEBlock(self.cfg) for _ in range(self.cfg.layers)])
        self.norm = RMSNorm(self.cfg.dim)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.cfg.context:
            raise ValueError('Sequence exceeds configured context')
        x = self.embedding(tokens)
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux = block(x)
            total_aux_loss = total_aux_loss + aux
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, total_aux_loss
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss, total_aux_loss

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
        print(f"⚠️ {MODEL_INPUT} não encontrado! Procurando localmente...")
        local_cand = list(Path(".").rglob("cerebro-moe-500m-continuo.pt"))
        if local_cand:
            model_path = local_cand[0]
        else:
            raise FileNotFoundError("Modelo base não encontrado!")
    else:
        model_path = MODEL_INPUT

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Extrair state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            raw_state = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint:
            raw_state = checkpoint["model_state"]
        elif "model" in checkpoint:
            raw_state = checkpoint["model"]
        else:
            raw_state = checkpoint
    else:
        raw_state = checkpoint

    # Limpar prefixos caso tenha sido salvo via torch.compile
    state_dict = {}
    for k, v in raw_state.items():
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = v
        else:
            state_dict[k] = v
    
    model = BrainMoE(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    print("✅ Cérebro MoE pronto!")

    # Dataset e DataLoader
    dataset = MaskedChatDataset(DATASET_FILE, tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    epochs = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=1e-5)

    print("\n" + "=" * 70)
    print(f"⚡ INICIANDO CALIBRAÇÃO GOLD COM MÁSCARA -100 ({epochs} Épocas | LR: 4e-5 -> 1e-5)...")
    print("=" * 70)

    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        steps = 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward com cálculo interno de loss com máscara -100 e aux_loss
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
        current_lr = scheduler.get_last_lr()[0]
        print(f"Época [{ep}/{epochs}] | Loss nas Respostas: {avg_loss:.4f} | LR: {current_lr:.2e} | Tempo: {dt:.2f}s")

    # Salvar Checkpoint
    print(f"\n💾 Salvando checkpoint calibrado em: {MODEL_OUTPUT}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_state": model.state_dict(),
        "config": cfg,
        "type": "Sparse-MoE-8x-Top2"
    }, MODEL_OUTPUT)
    print("✅ Checkpoint salvo com sucesso no Google Drive!")

    # Chat de Verificação dos Testes de Ouro
    print("\n" + "=" * 70)
    print("📋 EXECUTANDO BATERIA DE HOMOLOGAÇÃO (TESTES DE OURO)")
    print("=" * 70)
    model.eval()

    def responder(pergunta):
        prompt = f"### Usuário:\n{pergunta}\n\n### Assistente:\n"
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
        generated = input_ids[0].tolist()
        
        with torch.no_grad():
            for _ in range(256):
                curr = torch.tensor([generated[-cfg.context:]], device=device)
                logits, _ = model(curr)
                next_token_logits = logits[0, -1, :].clone()
                next_token_logits[pad_token_id] = -float('inf')
                
                # Penalidade de repetição para evitar loops
                for tok_id in set(generated[-30:]):
                    if next_token_logits[tok_id] > 0:
                        next_token_logits[tok_id] /= 1.15
                    else:
                        next_token_logits[tok_id] *= 1.15
                
                # Amostragem Top-K (40) com temperatura 0.6
                top_k = 40
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('inf')
                
                probs = F.softmax(next_token_logits / 0.6, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
                if nxt == (tokenizer.token_to_id("<eos>") or 2):
                    break
                generated.append(nxt)
                
        resposta_txt = tokenizer.decode(generated[len(input_ids[0]):])
        return resposta_txt.strip()

    testes = [
        "oi",
        "quem é você e qual é o seu objetivo em uma frase simples?",
        "preciso de sua ajuda",
        "quanto é 29 + 20?",
        "Se eu tenho 3 maçãs e como 1, depois ganho mais 5, com quantas maçãs eu fico? Mostre apenas a conta.",
        "Como fazer um loop for em Python?",
        "Escreva uma função em Python que soma dois números",
        "Qual a diferença entre uma lista e uma tupla em Python?",
        "Explique em duas frases o que é a gravidade.",
        "Me dê uma lista de 3 frutas amarelas.",
        "como eu faço novas amizades?",
        "eu preciso de uma dica sua"
    ]
    for q in testes:
        print(f"\nVocê: {q}")
        resp = responder(q)
        print(f"🤖 Synapse:\n{resp}\n{'-'*50}")

    print("\n" + "=" * 70)
    print("💬 SYNAPSE ONLINE — CHAT LIVRE (Digite 'sair' para encerrar)")
    print("=" * 70)
    while True:
        try:
            q = input("\nVocê: ")
            if q.strip().lower() in ["sair", "exit", "quit", "q"]:
                print("Encerrando sessão interativa.")
                break
            if not q.strip():
                continue
            resp = responder(q)
            print(f"\n🤖 Synapse:\n{resp}\n{'-'*50}")
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main()
