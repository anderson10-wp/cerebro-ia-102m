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

print(f"{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}{CYAN}🧠 SYNAPSE MoE 500M — CHAT INTERATIVO (RX 580 TURBO DIRECTML){RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}")

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
WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
DOWNLOADS = Path(os.path.expanduser(r"~\Downloads"))

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

try:
    from modelo_synapse_1b import BrainMoE1B, Config1B
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Config1B])
except Exception:
    pass

CHECKPOINT_CANDIDATES = [
    WORKSPACE / "synapse-1b-r1-final.pt",
    DOWNLOADS / "synapse-1b-r1-final.pt",
    WORKSPACE / "synapse-1b-latest.pt",
    DOWNLOADS / "synapse-1b-latest.pt",
    WORKSPACE / "cerebro-moe-500m-synapse.pt",
    DOWNLOADS / "cerebro-moe-500m-synapse.pt",
    Path("synapse-1b-r1-final.pt"),
    Path("cerebro-moe-500m-synapse.pt"),
]

CKPT_FILE = None
for cp in CHECKPOINT_CANDIDATES:
    if cp.exists():
        CKPT_FILE = str(cp)
        break

if not CKPT_FILE:
    print(f"\n{YELLOW}❌ Nenhum checkpoint suportado encontrado!{RESET}")
    print(f"Baixe o arquivo do seu Google Drive (ex: synapse-1b-r1-final.pt) e coloque em:")
    print(f" -> {DOWNLOADS / 'synapse-1b-r1-final.pt'}")
    print(f" ou em:")
    print(f" -> {WORKSPACE / 'synapse-1b-r1-final.pt'}")
    sys.exit(1)

TOK_CANDIDATES = [
    Path("tokenizer.json"),
    Path("/content/cerebro-ia-102m/tokenizer.json"),
    WORKSPACE / "tokenizer.json",
    WORKSPACE / "outputs" / "h100-pretrain" / "result" / "tokenizer.json",
]

TOK_FILE = None
for tp in TOK_CANDIDATES:
    if tp.exists():
        TOK_FILE = str(tp)
        break

if not TOK_FILE:
    cand = list(Path(".").rglob("tokenizer.json"))
    if not cand and WORKSPACE.exists():
        cand = list(WORKSPACE.rglob("tokenizer.json"))
    if cand:
        TOK_FILE = str(cand[0])
    else:
        print(f"{YELLOW}❌ Tokenizador não encontrado!{RESET}")
        sys.exit(1)

# 3. Definição da Arquitetura MoE 500M Oficial (RoPE + QKV Fundido + KV-Cache)
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
import __main__
__main__.Config = Config

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

# 4. Carregamento dos Pesos e Detecção de Arquitetura
print(f"📦 Carregando modelo ({Path(CKPT_FILE).name})...")
t0 = time.time()
raw = torch.load(CKPT_FILE, map_location='cpu', weights_only=False)

if isinstance(raw, dict):
    state = raw.get("model_state_dict", raw.get("model_state", raw.get("model", raw)))
else:
    state = raw

# Remove prefixo de compilação se houver
cleaned_state = {}
for k, v in state.items():
    if k.startswith("_orig_mod."):
        cleaned_state[k[len("_orig_mod."):]] = v
    else:
        cleaned_state[k] = v

dim = cleaned_state["embedding.weight"].shape[1]
if dim == 768:
    from modelo_synapse_1b import BrainMoE1B, Config1B
    cfg = Config1B()
    model = BrainMoE1B(cfg)
    nome_modelo = "Synapse 1.2B MoE (Top-2 R1)"
else:
    cfg = Config()
    model = BrainMoE(cfg)
    nome_modelo = "Synapse MoE 500M"

model.load_state_dict(cleaned_state)
model = model.to(device)
model.eval()
print(f"{GREEN}✅ {nome_modelo} pronto na memória ({device_name}) em {time.time()-t0:.2f}s!{RESET}")

from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOK_FILE)
eos_id = tok.token_to_id("<eos>") or 2
pad_id = tok.token_to_id("<pad>") or 0
print(f"{GREEN}✅ Tokenizador pronto!{RESET}")

print(f"\n{BOLD}{CYAN}💬 SYNAPSE ONLINE — PRONTA PARA CONVERSAR{RESET}")
print(f"{DIM}Identidade: Synapse | Criador: Nandinho | Hardware: AMD Radeon RX 580{RESET}")
print(f"{YELLOW}Dica: Digite sua mensagem e pressione ENTER. Digite 'sair' para encerrar.{RESET}\n")

while True:
    try:
        user_input = input(f"{BOLD}{GREEN}Você: {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAté logo!")
        break
        
    if not user_input:
        continue
    if user_input.lower() in ["sair", "exit", "quit"]:
        print(f"\n{CYAN}Encerrando sessão com a Synapse. Até breve!{RESET}")
        break
        
    # Template calibrado no SFT
    prompt = f"### Usuário:\n{user_input}\n\n### Assistente:\n"
    prompt_ids = tok.encode(prompt).ids
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    print(f"\n{BOLD}{CYAN}🤖 Synapse:{RESET} ", end="", flush=True)
    
    in_think = False
    generated_tokens = list(prompt_ids)
    
    with torch.no_grad():
        # Prefill rápido com KV-Cache
        logits, kv_caches = model(input_tensor, kv_caches=[(None, None)]*cfg.layers, start_pos=0)
        next_token_logits = logits[0, -1, :].clone()
        next_token_logits[pad_id] = -float('inf')
        
        # Top-K (40) e temperatura 0.6
        top_k = 40
        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
        next_token_logits[next_token_logits < v[-1]] = -float('inf')
        probs = F.softmax(next_token_logits / 0.6, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        first_id = next_token.item()
        generated_tokens.append(first_id)
        
        first_str = tok.decode([first_id])
        if "<think>" in first_str:
            in_think = True
            sys.stdout.write(f"\n{DIM}{YELLOW}<think>{RESET}\n")
        else:
            sys.stdout.write(first_str)
        sys.stdout.flush()
        
        # Decoding Token a Token
        curr_token_tensor = next_token.unsqueeze(0)
        curr_pos = len(prompt_ids)
        
        for step in range(512):
            logits, kv_caches = model(curr_token_tensor, kv_caches=kv_caches, start_pos=curr_pos)
            next_token_logits = logits[0, -1, :].clone()
            next_token_logits[pad_id] = -float('inf')
            
            # Trava de Integridade: NÃO permitir <eos> prematuro enquanto estiver dentro do <think>
            if in_think:
                next_token_logits[eos_id] = -float('inf')
            
            # Penalidade de repetição leve
            for tok_id in set(generated_tokens[-25:]):
                if next_token_logits[tok_id] > 0:
                    next_token_logits[tok_id] /= 1.15
                else:
                    next_token_logits[tok_id] *= 1.15
                    
            # Top-K (40) com temp 0.6
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[-1]] = -float('inf')
            probs = F.softmax(next_token_logits / 0.6, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            
            if token_id == eos_id and not in_think:
                break
                
            generated_tokens.append(token_id)
            token_str = tok.decode([token_id])
            accum_str = tok.decode(generated_tokens[len(prompt_ids):])
            
            # Atualiza estado do pensamento
            was_in_think = in_think
            if "<think>" in accum_str and "</think>" not in accum_str:
                in_think = True
            elif "</think>" in accum_str:
                in_think = False
            
            # Formatação visual fluida no terminal (suprime fragmentos de tags)
            if "<think>" in token_str:
                sys.stdout.write(f"\n{DIM}{YELLOW}<think>{RESET}\n")
            elif was_in_think and not in_think:
                sys.stdout.write(f"\n{DIM}{YELLOW}</think>{RESET}\n\n")
            elif "</" in token_str or "think>" in token_str or "<think" in token_str:
                pass
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
                
    print("\n" + "-" * 70 + "\n")
