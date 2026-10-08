"""Generate text with a trained checkpoint. This is a base model, not a chat assistant."""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Brain, Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--prompt', default='A inteligência artificial pode')
    p.add_argument('--tokens', type=int, default=100)
    args = p.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    net = Brain(Config(**saved['config']))
    net.load_state_dict(saved['model'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt).ids
    if not ids:
        raise ValueError('Prompt is empty')
    x = torch.tensor([ids], device=device)
    z = net.generate(x, new_tokens=args.tokens)
    print(tok.decode(z[0].tolist()))

if __name__ == '__main__':
    main()
