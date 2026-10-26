import os, sys, math, time
from dataclasses import dataclass
from pathlib import Path

# Enable UTF-8 for console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Terminal ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

print(f"{BOLD}{MAGENTA}{'='*70}{RESET}")
print(f"{BOLD}{MAGENTA}🧠 CÉREBRO MoE 500M — CHAT INTERATIVO (MODO PENSAR + KV-CACHE TURBO){RESET}")
print(f"{BOLD}{MAGENTA}{'='*70}{RESET}")

# 1. Hardware Detection: AMD Radeon RX 580 (DirectML) or CPU
device_name = "CPU"
device = "cpu"

try:
    import torch_directml
    device = torch_directml.device()
    device_name = torch_directml.device_name(0)
    print(f"{GREEN}🚀 Placa de Vídeo Detectada: {device_name} (DirectML Ativo){RESET}")
except Exception as e:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"{GREEN}🚀 GPU Detectada: {device_name} (CUDA Ativo){RESET}")
    else:
        device = torch.device("cpu")
        print(f"{YELLOW}⚡ Rodando em: Processador CPU (DirectML não disponível){RESET}")

import torch
import torch.nn as nn
import torch.nn.functional as F

# 2. Localização dos Pesos e Tokenizador
CHECKPOINT_PATHS = [
    os.path.expanduser(r"~\Downloads\cerebro-moe-500m-modo-pensar.pt"),
    r"C:\Users\Fernando\Downloads\cerebro-moe-500m-modo-pensar.pt",
    "cerebro-moe-500m-modo-pensar.pt",
    r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m\cerebro-moe-500m-modo-pensar.pt"
]

CKPT_FILE = None
for cp in CHECKPOINT_PATHS:
    if os.path.exists(cp):
        CKPT_FILE = cp
        break

if not CKPT_FILE:
    print(f"{YELLOW}❌ Arquivo 'cerebro-moe-500m-modo-pensar.pt' não encontrado!{RESET}")
    sys.exit(1)

TOK_PATHS = [
    r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m\outputs\h100-pretrain\result\tokenizer.json",
    "tokenizer.json",
    "work/kaggle-sft-data/tokenizer.json"
]

TOK_FILE = None
for tp in TOK_PATHS:
    if os.path.exists(tp):
        TOK_FILE = tp
        break

if not TOK_FILE:
    tok_candidates = list(Path(".").rglob("tokenizer.json"))
    if tok_candidates:
        TOK_FILE = str(tok_candidates[0])

# 3. Definição da Arquitetura MoE 500M com KV-Cache
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

cfg = Config()

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
        assert cfg.dim % cfg.heads == 0 and self.head_dim % 2 == 0
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        inv = cfg.rope_base ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        angles = torch.outer(torch.arange(cfg.context).float(), inv)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def rotate(self, x, offset=0):
        length = x.size(-2)
        cos = self.cos[offset:offset+length].to(x.dtype)[None, None]
        sin = self.sin[offset:offset+length].to(x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x, kv_cache=None, start_pos=0):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q = self.rotate(q, start_pos)
        k = self.rotate(k, start_pos)
        
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            if cached_k is not None:
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        is_causal = (t > 1)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        return self.out(z.transpose(1, 2).contiguous().view(b, t, d)), new_cache

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
                out[token_indices] = out[token_indices] + (expert_out * weights).to(out.dtype)
        
        return out.view(b, t, d)

class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.moe = MoELayer(cfg)

    def forward(self, x, kv_cache=None, start_pos=0):
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out
        x = x + self.moe(self.norm2(x))
        return x, new_cache

class BrainMoE(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
        self.blocks = nn.ModuleList([MoEBlock(self.cfg) for _ in range(self.cfg.layers)])
        self.norm = RMSNorm(self.cfg.dim)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens, kv_caches=None, start_pos=0):
        x = self.embedding(tokens)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=cache_i, start_pos=start_pos)
            new_caches.append(new_cache)
        logits = self.head(self.norm(x))
        return logits, new_caches

# 4. Carregamento dos Pesos
print(f"📦 Carregando modelo ({CKPT_FILE})...")
model = BrainMoE(cfg)
t0 = time.time()
raw = torch.load(CKPT_FILE, map_location='cpu', weights_only=False)
state = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
model.load_state_dict(state)
model = model.to(device)
model.eval()
print(f"{GREEN}✅ Modelo MoE 500M pronto na memória ({device_name}) em {time.time()-t0:.2f}s!{RESET}")

from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOK_FILE)
print(f"{GREEN}✅ Tokenizador pronto!{RESET}")

SYSTEM_PROMPT = (
    "Você é o Cérebro IA, um modelo de linguagem avançado com 500 milhões de parâmetros "
    "operando em arquitetura Mixture of Experts (MoE) com 8 especialistas e Modo Pensar ativo. "
    "Sua missão é responder com clareza, inteligência e profundidade, raciocinando passo a passo "
    "dentro de tags <think>...</think> antes de fornecer sua resposta final em português."
)

print(f"\n{BOLD}{CYAN}💬 System Prompt Ativo:{RESET}")
print(f"{DIM}{SYSTEM_PROMPT}{RESET}\n")
print(f"{YELLOW}💡 Dica: Digite sua pergunta e pressione ENTER. Digite 'sair' para encerrar.{RESET}\n")

while True:
    try:
        user_input = input(f"{BOLD}{GREEN}Você: {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAté logo!")
        break
        
    if not user_input:
        continue
    if user_input.lower() in ["sair", "exit", "quit"]:
        print(f"\n{MAGENTA}Encerrando sessão com o Cérebro IA. Até breve!{RESET}")
        break
        
    prompt = f"Sistema: {SYSTEM_PROMPT}\n\nPergunta: {user_input}\nResposta: <think>\n"
    prompt_ids = tok.encode(prompt).ids
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    print(f"\n{BOLD}{MAGENTA}Cérebro IA:{RESET} ", end="", flush=True)
    
    in_think = True
    generated_tokens = list(prompt_ids)
    
    # 1. Prefill com KV-Cache (rápido!)
    with torch.no_grad():
        logits, kv_caches = model(input_tensor, kv_caches=[(None, None)]*cfg.layers, start_pos=0)
        next_token = torch.multinomial(F.softmax(logits[0, -1, :] / 0.65, dim=-1), num_samples=1)
        generated_tokens.append(next_token.item())
        
        # Stream do primeiro token
        first_token_str = tok.decode([next_token.item()])
        sys.stdout.write(f"{YELLOW}{first_token_str}{RESET}")
        sys.stdout.flush()
        
        # 2. Decoding Token a Token com KV-Cache acelerado O(1)
        curr_token_tensor = next_token.unsqueeze(0)
        curr_pos = len(prompt_ids)
        
        for _ in range(250):
            logits, kv_caches = model(curr_token_tensor, kv_caches=kv_caches, start_pos=curr_pos)
            next_token = torch.multinomial(F.softmax(logits[0, -1, :] / 0.65, dim=-1), num_samples=1)
            token_id = next_token.item()
            generated_tokens.append(token_id)
            
            token_str = tok.decode([token_id])
            if "</think>" in token_str:
                in_think = False
                sys.stdout.write(f"\n{BOLD}Resposta:{RESET}\n")
            else:
                if in_think:
                    sys.stdout.write(f"{YELLOW}{token_str}{RESET}")
                else:
                    sys.stdout.write(f"{token_str}")
            sys.stdout.flush()
            
            curr_token_tensor = next_token.unsqueeze(0)
            curr_pos += 1
            
            if curr_pos >= cfg.context:
                break
                
    print("\n" + "-" * 70)
