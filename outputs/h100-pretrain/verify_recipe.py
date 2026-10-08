"""CPU verification of gradient accumulation and checkpoint/optimizer migration."""
import copy, json, tempfile
from pathlib import Path
import torch
from model import Brain, Config
from training_helpers import atomic_save
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2); torch.manual_seed(6)
cfg=Config(vocab_size=64,dim=16,layers=1,heads=2,hidden_dim=32,context=8)
a=Brain(cfg)
def optimizer(model):
    return torch.optim.AdamW([
        {'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.1},
        {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0}],lr=3e-5,betas=(.9,.95))
oa=optimizer(a)
x=torch.randint(0,64,(4,8)); y=torch.randint(0,64,(4,8))
a(x,y).backward(); oa.step(); oa.zero_grad(set_to_none=True)
with tempfile.TemporaryDirectory() as folder:
    checkpoint=Path(folder)/'source.pt'
    atomic_save({'model':a.state_dict(),'optimizer':oa.state_dict()},checkpoint)
    saved=torch.load(checkpoint,weights_only=True)
    b=Brain(cfg); b.load_state_dict(saved['model']); ob=optimizer(b); ob.load_state_dict(saved['optimizer'])
    if b.embedding.weight is not b.head.weight: raise RuntimeError('Weight tying lost')
    a(x,y).backward()
    for i in range(2): (b(x[i*2:(i+1)*2],y[i*2:(i+1)*2])/2).backward()
    torch.nn.utils.clip_grad_norm_(a.parameters(),1.); torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
    oa.step(); ob.step()
    delta=max((p-q).abs().max().item() for p,q in zip(a.parameters(),b.parameters()))
    if delta>1e-7: raise RuntimeError('Migration or accumulation changed the update')
    result={'status':'passed','max_parameter_difference':delta,'optimizer_migration_verified':True,
            'gradient_accumulation_verified':True,'weight_tying_verified':True,
            'gpu_precision_and_performance_tested':False,'note':'Small CPU fixture; H100 BF16 and throughput require the actual GPU run.'}
    (ROOT/'recipe-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
