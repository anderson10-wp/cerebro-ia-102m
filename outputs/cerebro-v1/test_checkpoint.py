"""Small reproducible CPU generation probe; does not update weights."""
import json
import argparse
import hashlib
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Brain, Config

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--final', action='store_true')
args = parser.parse_args()
folder = root / ('trained' if args.final else 'test-74') / 'brain'
result_folder = root / ('test-final' if args.final else 'test-74')
result_folder.mkdir(exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(42)
saved = torch.load(folder / 'best-model.pt', map_location='cpu', weights_only=False)
net = Brain(Config(**saved['config']))
net.load_state_dict(saved['model'])
net.eval()
tokenizer_path = folder / 'tokenizer.json'
if args.final and not tokenizer_path.exists():
    tokenizer_path = root / 'test-74' / 'brain' / 'tokenizer.json'
    manifest = json.loads((folder / 'data-manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == manifest['sha256']['tokenizer.json']
tok = Tokenizer.from_file(str(tokenizer_path))
prompts = [
    'O Brasil é um país localizado na',
    'A inteligência artificial pode ajudar as pessoas a',
    'Pergunta: Olá! Como você está?\nResposta:',
    'Pergunta: Quanto é 2 + 2?\nResposta:',
    'def soma(a, b):\n    ',
]
results = []
for prompt in prompts:
    started = time.perf_counter()
    ids = torch.tensor([tok.encode(prompt).ids])
    output = net.generate(ids, new_tokens=48, temperature=0.7, top_k=40)
    item = {'prompt': prompt, 'continuation': tok.decode(output[0, ids.shape[1]:].tolist()),
            'seconds': round(time.perf_counter() - started, 2)}
    results.append(item)
    print(json.dumps(item, ensure_ascii=True), flush=True)
    (result_folder / 'generation-results.json').write_text(
        json.dumps({'checkpoint': ('session-03' if args.final else 'session-02') + ' best-model.pt', 'seed': 42,
                    'new_tokens': 48, 'temperature': 0.7, 'top_k': 40,
                    'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
