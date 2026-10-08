"""Private bounded CUDA SFT; no credentials, tools or automatic promotion."""
import hashlib,json,math,random,shutil,sys,time,unicodedata
from dataclasses import asdict
from pathlib import Path
import torch
from tokenizers import Tokenizer

def read(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    roots=[p.parent for p in Path('/kaggle/input').rglob('manifest.json') if (p.parent/'base.pt').exists()]
    assert len(roots)==1, roots
    data=roots[0]; out=Path('/kaggle/working/brain-sft'); out.mkdir(exist_ok=True)
    meta=json.loads((data/'manifest.json').read_text())
    for name,digest in meta['files'].items(): assert sha(data/name)==digest,name
    sys.path.insert(0,str(data))
    from model import Brain,Config
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU training disabled')
    torch.set_num_threads(4); torch.manual_seed(8181); random.seed(8181)
    saved=torch.load(data/'base.pt',map_location='cpu',weights_only=True)
    net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model']); net.cuda()
    tok=Tokenizer.from_file(str(data/'tokenizer.json')); pad=tok.token_to_id('<|pad|>'); eos=tok.token_to_id('<|eos|>')
    def encode(rows):
        result=[]
        for r in rows:
            p=tok.encode(r['prompt'],add_special_tokens=False).ids
            a=tok.encode(r['response']+'<|eos|>',add_special_tokens=False).ids
            assert p and a[-1]==eos and len(p+a)<=net.cfg.context
            result.append((p+a[:-1],[-100]*(len(p)-1)+a))
        return result
    train=encode(read(data/'train.jsonl')); dev=encode(read(data/'dev.jsonl'))
    def batch(rows):
        n=max(len(x) for x,y in rows)
        x=torch.full((len(rows),n),pad,dtype=torch.long)
        y=torch.full_like(x,-100)
        for i,(a,b) in enumerate(rows): x[i,:len(a)]=torch.tensor(a); y[i,:len(b)]=torch.tensor(b)
        return x.cuda(),y.cuda()
    class WeightedLoss(torch.nn.Module):
        def __init__(self,model): super().__init__(); self.model=model
        def forward(self,x,y):
            with torch.autocast('cuda',dtype=torch.float16): loss=self.model(x,y)
            count=(y!=-100).sum()
            return torch.stack((loss*count,count))
    parallel=torch.nn.DataParallel(WeightedLoss(net)) if torch.cuda.device_count()>1 else WeightedLoss(net)
    # A row per replica preserves token weighting for unequal sequence lengths.
    def loss_for(x,y):
        values=parallel(x,y).reshape(-1,2)
        return values[:,0].sum()/values[:,1].sum()
    @torch.no_grad()
    def evaluate(rows):
        net.eval(); total=0.; count=0
        for start in range(0,len(rows),8):
            x,y=batch(rows[start:start+8]); loss=loss_for(x,y); n=int((y!=-100).sum())
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite validation loss')
            total+=loss.item()*n; count+=n
        net.train(); return total/count
    @torch.no_grad()
    def answers():
        net.eval(); results=[]; pairs={}
        for r in read(data/'tests.jsonl'):
            x=torch.tensor([tok.encode(r['prompt']).ids],device='cuda'); generated=[]
            for _ in range(48):
                with torch.autocast('cuda',dtype=torch.float16): value=net(x)[:,-1].argmax(-1,keepdim=True)
                if value.item()==eos: break
                generated.append(value.item()); x=torch.cat((x,value),1)
            text=tok.decode(generated).strip()
            if r['scorer']=='exact': ok=unicodedata.normalize('NFC',text)==unicodedata.normalize('NFC',r['response'].strip())
            else:
                try: ok=json.dumps(json.loads(text),sort_keys=True,allow_nan=False)==json.dumps(json.loads(r['response']),sort_keys=True,allow_nan=False)
                except (ValueError,TypeError): ok=False
            results.append({'id':r['id'],'output':text,'expected':r['response'],'correct':ok})
            if 'pair_id' in r: pairs.setdefault(r['pair_id'],[]).append(ok)
        net.train()
        return {'correct':sum(r['correct'] for r in results),'total':len(results),'correct_pairs':sum(len(v)==2 and all(v) for v in pairs.values()),'total_pairs':len(pairs),'results':results}
    opt=torch.optim.AdamW(net.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    report={'status':'initial_evaluation','gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'manifest':meta,'steps':0,'response_tokens_seen':0,'automatic_promotion':False,'validation':[]}
    def log():
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ['manifest','before','after','validation']},ensure_ascii=False),flush=True)
    def checkpoint(name,epoch,cursor,order):
        temp=out/(name+'.tmp')
        torch.save({'config':asdict(net.cfg),'model':{k:v.detach().cpu().half() for k,v in net.state_dict().items()},'optimizer':opt.state_dict() if name=='last.pt' else None,'scaler':scaler.state_dict(),'epoch':epoch,'cursor':cursor,'order':order if name=='last.pt' else None,'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'sft_steps':report['steps'],'base_pretraining_tokens':saved.get('base_pretraining_tokens',saved.get('tokens')),'validation_loss':report.get('best_validation_loss'),'manifest_sha256':sha(data/'manifest.json')},temp)
        temp.replace(out/name)
    shutil.copyfile(data/'tokenizer.json',out/'tokenizer.json'); shutil.copyfile(data/'model.py',out/'model.py')
    shutil.copyfile(data/'manifest.json',out/'data-manifest.json')
    log(); initial=evaluate(dev); report['initial_validation_loss']=initial; report['best_validation_loss']=initial
    report['before']=answers(); log()
    # Best begins with the original weights: a worse candidate never replaces them.
    shutil.copyfile(data/'base.pt',out/'best-model.pt')
    start=time.monotonic(); failures=0; overflows=0; stop=False
    try:
        for epoch in range(2):
            order=list(range(len(train))); random.shuffle(order)
            # Local length bucketing reduces padding without sorting the entire epoch.
            for start_idx in range(0,len(order),256): order[start_idx:start_idx+256]=sorted(order[start_idx:start_idx+256],key=lambda i:len(train[i][0]))
            for cursor in range(0,len(order),8):
                if time.monotonic()-start>=4500: stop=True; break
                x,y=batch([train[i] for i in order[cursor:cursor+8]])
                opt.zero_grad(set_to_none=True); loss=loss_for(x,y)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                if not torch.isfinite(norm):
                    scaler.update(new_scale=scaler.get_scale()/2); overflows+=1
                    if overflows>16: raise RuntimeError('Repeated gradient overflow')
                    continue
                scaler.step(opt); scaler.update(); overflows=0
                report['steps']+=1; report['response_tokens_seen']+=int((y!=-100).sum())
                if report['steps']%25==0:
                    report.update(status='training',epoch=epoch+1,loss=loss.item(),elapsed_seconds=round(time.monotonic()-start)); log()
                if report['steps']%250==0: checkpoint('last.pt',epoch,cursor+8,order)
            current=evaluate(dev); report['validation'].append({'epoch':epoch+1,'loss':current,'steps':report['steps']})
            if current<report['best_validation_loss']:
                report['best_validation_loss']=current; failures=0; checkpoint('best-model.pt',epoch,cursor+8,order)
            else: failures+=1
            checkpoint('last.pt',epoch,cursor+8,order); log()
            if failures>=1 or stop: break
        best=torch.load(out/'best-model.pt',map_location='cpu',weights_only=True); net.load_state_dict(best['model'])
        report['after']=answers(); report['audit_loss']=evaluate(encode(read(data/'audit.jsonl')))
        report.update(status='completed_candidate_only',elapsed_seconds=round(time.monotonic()-start),selected_checkpoint_sha256=sha(out/'best-model.pt'))
        log()
    except BaseException:
        report.update(status='failed',elapsed_seconds=round(time.monotonic()-start)); log(); raise
if __name__=='__main__': main()
