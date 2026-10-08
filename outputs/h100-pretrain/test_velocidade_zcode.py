"""Medicao de velocidade de geracao na CPU (teste do ZCode)."""
import time
import torch
from tokenizers import Tokenizer
from model import Brain, Config

torch.set_num_threads(4)
saved = torch.load('result/best-model.pt', map_location='cpu', weights_only=True)
net = Brain(Config(**saved['config']))
net.load_state_dict(saved['model'])
net.eval()
print('parametros:', net.parameter_count())

tok = Tokenizer.from_file('result/tokenizer.json')
prompt = tok.encode('Pergunta: Escreva um paragrafo sobre o mar.\nResposta:').ids
x = torch.tensor([prompt])
n = 100
t0 = time.time()
with torch.no_grad():
    for _ in range(n):
        logits = net(x[:, -net.cfg.context:])[:, -1]
        x = torch.cat([x, logits.argmax(-1, keepdim=True)], 1)
dt = time.time() - t0
print(f'{n} tokens em {dt:.1f}s = {n/dt:.2f} tokens/s (CPU, 4 threads, batch=1)')
