"""Single bounded pilot; compare frozen language test and retention, candidate only."""
import hashlib,json,random,sys,time,unicodedata
from dataclasses import asdict
from pathlib import Path
import torch
from tokenizers import Tokenizer

def score(text,r):
    if r['scorer']=='manual': return None
    if r['scorer']=='exact': return unicodedata.normalize('NFC',text.strip())==unicodedata.normalize('NFC',r['response'].strip())
    try: return json.dumps(json.loads(text),sort_keys=True,allow_nan=False)==json.dumps(json.loads(r['response']),sort_keys=True,allow_nan=False)
    except (ValueError,TypeError): return False

def main():
    roots=[p.parent for p in Path('/kaggle/input').rglob('base.pt') if (p.parent/'model.py').exists()]
    assert len(roots)==1; data=roots[0]; sys.path.insert(0,str(data)); from model import Brain,Config
    base=data/'base.pt'; assert hashlib.sha256(base.read_bytes()).hexdigest()=='1ea8da9fa900f7f2ec1beaf5f34511d449ea695732e19d1f7744d4efb21f9a9e'
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required, no CPU training fallback')
    torch.set_num_threads(4); torch.manual_seed(61820); rng=random.Random(61820)
    out=Path('/kaggle/working/language-pilot'); out.mkdir(exist_ok=True)
    saved=torch.load(base,map_location='cpu',weights_only=True)
    net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model']); net.cuda()
    tok=Tokenizer.from_file(str(data/'tokenizer.json')); eos=tok.token_to_id('<|eos|>'); pad=tok.token_to_id('<|pad|>')
    def encode(rows):
        result=[]
        for r in rows:
            pi=tok.encode(r['prompt']).ids; ai=tok.encode(r['response']+'<|eos|>').ids
            assert pi and ai[-1]==eos and len(pi+ai)<=384
            result.append((pi+ai[:-1],[-100]*(len(pi)-1)+ai))
        return result
    train=encode(DATA['train']+DATA['replay']); dev=encode(DATA['dev'])
    def batch(rows):
        n=max(len(x) for x,y in rows); x=torch.full((len(rows),n),pad,dtype=torch.long); y=torch.full_like(x,-100)
        for i,(a,b) in enumerate(rows): x[i,:len(a)]=torch.tensor(a); y[i,:len(b)]=torch.tensor(b)
        return x.cuda(),y.cuda()
    class WeightedLoss(torch.nn.Module):
        def __init__(self,model): super().__init__(); self.model=model
        def forward(self,x,y):
            with torch.autocast('cuda',dtype=torch.float16): loss=self.model(x,y)
            n=(y!=-100).sum(); return torch.stack((loss*n,n))
    wrapped=WeightedLoss(net); parallel=torch.nn.DataParallel(wrapped) if torch.cuda.device_count()>1 else wrapped
    def loss_for(x,y):
        v=parallel(x,y).reshape(-1,2); return v[:,0].sum()/v[:,1].sum()
    @torch.no_grad()
    def dev_loss():
        net.eval(); total=0.; count=0
        for start in range(0,len(dev),16):
            x,y=batch(dev[start:start+16]); n=int((y!=-100).sum()); v=loss_for(x,y)
            if not torch.isfinite(v): raise RuntimeError('Nonfinite validation')
            total+=v.item()*n; count+=n
        net.train(); return total/count
    report={'status':'initial_evaluation','gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'manifest':MANIFEST,'api_teacher_calls':0,'automatic_promotion':False,'steps':0,'response_tokens_seen':0,'validation':[],'recipe':{'epochs':4,'seconds':600,'batch':16,'learning_rate':2e-5,'seed':61820},'limits':'Curated pilot, not evidence of broad autonomous capability. Open responses require manual review. Reused retention tests are development diagnostics.'}
    def log(event):
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(event,ensure_ascii=False),flush=True)
    @torch.no_grad()
    def evaluate(rows,label):
        net.eval(); results=[]; pairs={}
        for i,r in enumerate(rows):
            x=torch.tensor([tok.encode(r['prompt']).ids],device='cuda'); generated=[]
            for _ in range(80 if r['scorer']=='manual' else 48):
                with torch.autocast('cuda',dtype=torch.float16): value=net(x)[:,-1].argmax(-1,keepdim=True)
                if value.item()==eos: break
                generated.append(value.item()); x=torch.cat((x,value),1)
            text=tok.decode(generated); correct=score(text,r)
            results.append({**r,'output':text,'correct':correct})
            if 'pair_id' in r: pairs.setdefault(r['pair_id'],[]).append((r['response'],correct))
            if (i+1)%24==0: log({'event':'evaluation','label':label,'done':i+1})
        scored=[r for r in results if r['correct'] is not None]
        pairs=[v for v in pairs.values() if len(v)==2 and v[0][0]!=v[1][0]]
        summary={'correct':sum(r['correct'] for r in scored),'total':len(scored),'correct_pairs':sum(all(ok for a,ok in v) for v in pairs),'total_pairs':len(pairs),'by_category':{c:{'correct':sum(r['correct'] for r in scored if r['category']==c),'total':sum(r['category']==c for r in scored)} for c in sorted({r['category'] for r in scored})},'results':results}
        net.train(); return summary
    initial=dev_loss(); report['initial_dev_loss']=initial; report['best_dev_loss']=initial
    report['before']=evaluate(DATA['test'],'before'); report['retention_before']=evaluate(DATA['retention'],'retention-before')
    log({'event':'baseline','correct':report['before']['correct'],'total':report['before']['total'],'retention':report['retention_before']['correct']})
    opt=torch.optim.AdamW(net.parameters(),lr=2e-5,betas=(.9,.95),weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024); best=initial; start=time.monotonic(); failures=0; overflow=0
    def save(name):
        tmp=out/(name+'.tmp')
        torch.save({'config':asdict(net.cfg),'model':{k:v.detach().cpu().half() for k,v in net.state_dict().items()},'steps':report['steps'],'validation_loss':best,'base_pretraining_tokens':saved.get('base_pretraining_tokens',saved.get('tokens')),'base_checkpoint_sha256':report['base_sha256']},tmp); tmp.replace(out/name)
    save('best-model.pt')
    for epoch in range(4):
        order=list(range(len(train))); rng.shuffle(order)
        for cursor in range(0,len(order),16):
            if time.monotonic()-start>=600: break
            x,y=batch([train[i] for i in order[cursor:cursor+16]])
            opt.zero_grad(set_to_none=True); loss=loss_for(x,y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite train loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not torch.isfinite(norm):
                scaler.update(new_scale=scaler.get_scale()/2); overflow+=1
                if overflow>16: raise RuntimeError('Repeated gradient overflow')
                continue
            scaler.step(opt); scaler.update(); overflow=0
            report['steps']+=1; report['response_tokens_seen']+=int((y!=-100).sum())
        current=dev_loss(); report['validation'].append({'epoch':epoch+1,'loss':current,'steps':report['steps']})
        if current<best: best=current; failures=0; report['best_dev_loss']=best; save('best-model.pt')
        else: failures+=1
        save('latest-model.pt'); log({'event':'epoch','epoch':epoch+1,'steps':report['steps'],'dev_loss':current})
        if failures>=2 or time.monotonic()-start>=600: break
    selected=torch.load(out/'best-model.pt',map_location='cpu',weights_only=True); net.load_state_dict(selected['model']); del selected
    report['after']=evaluate(DATA['test'],'after'); report['retention_after']=evaluate(DATA['retention'],'retention-after')
    old={r['id']:r['correct'] for r in report['retention_before']['results']}
    regressions=[r['id'] for r in report['retention_after']['results'] if old[r['id']] and not r['correct']]
    report.update(status='completed_candidate_only',elapsed_training_and_final_evaluation_seconds=round(time.monotonic()-start),selected_sha256=hashlib.sha256((out/'best-model.pt').read_bytes()).hexdigest(),retention_regressions=regressions,objective_gate_passed=report['after']['correct']>=report['before']['correct']+8 and not regressions,open_response_review_required=True,large_training_started=False)
    log({'event':'completed','before':report['before']['correct'],'after':report['after']['correct'],'total':report['after']['total'],'retention_before':report['retention_before']['correct'],'retention_after':report['retention_after']['correct'],'regressions':regressions,'objective_gate_passed':report['objective_gate_passed']})
if __name__=='__main__': main()
