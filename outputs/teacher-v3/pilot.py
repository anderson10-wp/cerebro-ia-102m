"""Default validates the recipe. --train explicitly runs a bounded local pilot."""
import argparse,hashlib,json,random,sys,time
from dataclasses import asdict
from pathlib import Path
import torch
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'agent-pilot'))
from sft import load_examples,batch,validation_loss
from model import Brain,Config
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,report):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(report,indent=2),encoding='utf-8'); temp.replace(path)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--train',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda','directml'],default='cpu')
    parser.add_argument('--out',type=Path,default=ROOT/'candidate')
    args=parser.parse_args()
    recipe=json.loads((ROOT/'pilot-config.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    assert digest(ROOT/'train.jsonl')==recipe['train_sha256']==manifest['train']['sha256']
    assert digest(ROOT/'validation.jsonl')==recipe['validation_sha256']==manifest['validation']['sha256']
    original=ROOT/recipe['base_checkpoint']; tokenizer=ROOT/recipe['tokenizer']
    assert digest(original)==recipe['base_checkpoint_sha256']
    assert digest(tokenizer)==manifest['tokenizer_sha256']
    torch.set_num_threads(2 if args.device=='directml' else 4); torch.manual_seed(42); rng=random.Random(42)
    tok=Tokenizer.from_file(str(tokenizer)); pad=tok.token_to_id('<|pad|>')
    saved=torch.load(original,map_location='cpu',weights_only=True)
    model=Brain(Config(**saved['config'])); model.load_state_dict(saved['model'])
    train=load_examples(ROOT/'train.jsonl',tok,model.cfg.context)
    dev=load_examples(ROOT/'validation.jsonl',tok,model.cfg.context)
    assert all(r['labels'][-1]==tok.token_to_id('<|eos|>') for r in train+dev)
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable; no cloud machine is started')
    device=torch.device('cpu')
    hardware='cpu'
    if args.device=='directml':
        sys.path.insert(0,str(ROOT.parent/'h100-pretrain'))
        from directml_backend import prepare_directml
        device,hardware=prepare_directml(model)
    else:
        device=torch.device(args.device); model.to(device)
    if not args.train:
        subset=[train[0],train[len(train)//2],train[-1]]
        x,y=batch(subset,pad,device); loss=model(x,y); loss.backward()
        assert torch.isfinite(loss.detach().cpu()) and all(torch.isfinite(p.grad.detach().cpu()).all() for p in model.parameters() if p.grad is not None)
        assert any(p.grad.detach().cpu().abs().max().item()>0 for p in model.parameters() if p.grad is not None)
        report={'status':'recipe_validated','training_examples':len(train),'validation_examples':len(dev),
            'masked_loss':loss.item(),'finite_backward':True,'nonzero_gradients':True,'eos_supervised':True,
            'optimizer_steps':0,'weights_written':False,'audit_examples_evaluated':0,'cloud_jobs_started':0,
            'device':str(device),'hardware':hardware}
        suffix='-directml' if args.device=='directml' else ''
        save(ROOT/('recipe-validation'+suffix+'.json'),report); print(json.dumps(report)); return
    out=args.out
    out.mkdir(exist_ok=False)
    opt=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False)
    # FP32 is intentional for this small local pilot.
    def dev_loss():
        model.eval(); total=0.; count=0
        with torch.no_grad():
            for offset in range(0,len(dev),recipe['batch_size']):
                x,y=batch(dev[offset:offset+recipe['batch_size']],pad,device)
                loss=model(x,y); n=int((y!=-100).sum())
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite validation')
                total+=loss.item()*n; count+=n
        model.train(); return total/count
    initial=dev_loss(); best=initial; failures=0; steps=0; tokens=0; start=time.monotonic()
    report={'status':'running','initial_validation_loss':initial,'base_checkpoint_sha256':recipe['base_checkpoint_sha256'],
        'validation':[],'automatic_promotion':False,'audit_examples_evaluated':0,'cloud_jobs_started':0,
        'device':str(device),'hardware':hardware}
    save(out/'report.json',report)
    try:
        for epoch in range(recipe['max_epochs']):
            order=list(train); rng.shuffle(order)
            for offset in range(0,len(order),recipe['batch_size']):
                if time.monotonic()-start>=recipe['max_training_seconds']: break
                x,y=batch(order[offset:offset+recipe['batch_size']],pad,device)
                opt.zero_grad(set_to_none=True); loss=model(x,y)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient')
                opt.step(); steps+=1; tokens+=int((y!=-100).sum())
            current=dev_loss(); report['validation'].append({'epoch':epoch+1,'loss':current,'steps':steps})
            if current<best:
                best=current; failures=0
                temp=out/'best-model.tmp'
                torch.save({'config':asdict(model.cfg),'model':{k:v.detach().cpu().half() for k,v in model.state_dict().items()},
                    'base_pretraining_tokens':saved.get('tokens',saved.get('base_pretraining_tokens')),'sft_steps':steps,'validation_loss':current},temp)
                temp.replace(out/'best-model.pt')
            else: failures+=1
            report.update(steps=steps,response_tokens_seen=tokens,best_validation_loss=best)
            save(out/'report.json',report)
            print(json.dumps(report['validation'][-1]),flush=True)
            if failures>=recipe['patience_epochs'] or time.monotonic()-start>=recipe['max_training_seconds']: break
        report.update(status='completed_candidate_only',candidate_saved=(out/'best-model.pt').exists())
    except BaseException:
        report.update(status='interrupted_or_failed'); raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,steps=steps,response_tokens_seen=tokens)
        save(out/'report.json',report)
if __name__=='__main__': main()
