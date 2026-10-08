"""Check ordinary text completion too, using greedy decoding on both models."""
import json,time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Brain,Config
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
prompts=['2 + 2 =','A capital da França é','O Brasil fica na América do','def add(a, b):\n    return']
results=[]
for name,folder in [('before',ROOT.parent/'cerebro-v1/trained/brain'),('after',ROOT/'result')]:
    torch.manual_seed(42)
    saved=torch.load(folder/'best-model.pt',map_location='cpu',weights_only=True)
    net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model']); net.eval()
    tok=Tokenizer.from_file(str(folder/'tokenizer.json'))
    for prompt in prompts:
        ids=torch.tensor([tok.encode(prompt).ids]); output=net.generate(ids,new_tokens=32,temperature=1,top_k=1)
        item={'model':name,'prompt':prompt,'continuation':tok.decode(output[0,ids.shape[1]:].tolist())}
        results.append(item); print(json.dumps(item,ensure_ascii=True),flush=True)
        (ROOT/'evaluation/completions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    del net,saved
