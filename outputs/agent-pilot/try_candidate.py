"""Local routing demonstration. Never executes any generated tool request."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from tokenizers import Tokenizer
from routing import ROOT, BRAIN, SYSTEM, TOOLS, validate_route

sys.path.insert(0, str(BRAIN))
from model import Brain, Config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request')
    parser.add_argument('--available', nargs='+', choices=TOOLS, default=TOOLS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.request.strip():
        parser.error('Request must not be empty')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    weights = ROOT/'lightning-result/best-model.pt'
    saved = torch.load(weights, map_location='cpu', weights_only=False)
    tok_path = BRAIN/'trained/brain/tokenizer.json'
    if hashlib.sha256(tok_path.read_bytes()).hexdigest() != saved['tokenizer_sha256']:
        raise RuntimeError('Candidate tokenizer mismatch')
    tok = Tokenizer.from_file(str(tok_path))
    net = Brain(Config(**saved['config']))
    net.load_state_dict(saved['model'])
    net.eval()
    prompt = '<|system|>'+SYSTEM+'\nDisponíveis: '+', '.join(args.available)+'<|user|>'+args.request+'<|assistant|>'
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids])
    if ids.shape[1] + 96 > net.cfg.context:
        raise ValueError('Request too long for the configured context')
    generated = []
    started = time.monotonic()
    with torch.no_grad():
        for _ in range(96):
            token = int(net(ids)[:, -1].argmax(-1).item())
            if token == tok.token_to_id('<|eos|>'):
                break
            generated.append(token)
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
            try:
                if isinstance(json.loads(tok.decode(generated)), dict):
                    break
            except ValueError:
                pass
    raw = tok.decode(generated)
    route, error = None, None
    try:
        route = validate_route(json.loads(raw), {'request': args.request, 'available': args.available})
    except (ValueError, TypeError) as exc:
        error = str(exc)
    result = {'request': args.request, 'available': args.available, 'raw_output': raw,
              'route': route, 'format_and_availability_valid': error is None, 'validation_error': error,
              'semantic_correctness_verified': False, 'tools_executed': 0,
              'seconds': round(time.monotonic()-started, 3), 'selected_training_step': saved['sft_steps'],
              'note': 'Experimental candidate. A valid structure can still contain the wrong tool or arguments.'}
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
