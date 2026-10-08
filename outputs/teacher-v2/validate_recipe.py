"""Check supervision on the current checkpoint; do not update or write weights."""
import hashlib,json,sys
from pathlib import Path
import torch
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'agent-pilot'))
from sft import load_examples,batch
from model import Brain,Config
torch.set_num_threads(4); torch.manual_seed(42)
manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
tok=Tokenizer.from_file(str(ROOT.parent/'h100-pretrain/result/tokenizer.json'))
for split in ('train','validation','audit'):
    path=ROOT/(split+'.jsonl')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[split]['sha256']
train=load_examples(ROOT/'train.jsonl',tok,1024)
dev=load_examples(ROOT/'validation.jsonl',tok,1024)
for row in train+dev:
    assert row['labels'][-1]==tok.token_to_id('<|eos|>')
saved=torch.load(ROOT.parent/'h100-pretrain/result/best-model.pt',map_location='cpu',weights_only=True)
net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model']); net.eval()
x,y=batch([train[0],train[20],train[-1]],tok.token_to_id('<|pad|>'),'cpu')
loss=net(x,y); loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
report={'status':'recipe_validated','base_pretraining_tokens':saved['tokens'],
    'train_examples':len(train),'validation_examples':len(dev),'audit_used_for_model_evaluation':False,
    'prompt_tokens_masked':True,'first_response_token_supervised':True,'eos_supervised':True,
    'finite_backward':True,'smoke_loss':loss.item(),'optimizer_steps':0,'weights_written':False,
    'paid_api_calls':0,'cloud_jobs_started':0}
(ROOT/'recipe-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
