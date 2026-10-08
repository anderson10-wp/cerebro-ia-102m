"""Controlled direct versus short verifiable comparison SFT, CUDA only."""
import hashlib,json,random,re,sys,time,unicodedata
from dataclasses import asdict
from pathlib import Path
import torch
from tokenizers import Tokenizer

def prompt(r,mode):
    instruction='Responda somente a resposta final.' if mode=='direct' else 'Mostre uma comparação curta verificável na linha Comparação: e depois a resposta na linha Final:.'
    return 'Pergunta: '+r['question']+' '+instruction+'\nResposta:'
def target(r,mode):
    return r['answer'] if mode=='direct' else 'Comparação: '+r['step']+'.\nFinal: '+r['answer']
def normalize(text): return unicodedata.normalize('NFC',text.strip())
def score(text,r,mode):
    final=text.strip()
    if mode=='reason':
        m=re.search(r'(?:^|\n)Final:\s*([^\n]+)',text)
        final=m.group(1).strip() if m else ''
    # Final labels have a fixed, case-sensitive command vocabulary.
    correct=normalize(final)==normalize(r['answer'])
    check=None
    if mode=='reason':
        m=re.search(r'(?:^|\n)Comparação:\s*([^\n]+)',text)
        check=bool(m and normalize(m.group(1)).rstrip('.')==normalize(r['step']))
    return correct,check

def main():
    root=next(p.parent for p in Path('/kaggle/input').rglob('base.pt') if (p.parent/'model.py').exists())
    assert hashlib.sha256((root/'base.pt').read_bytes()).hexdigest()=='1ea8da9fa900f7f2ec1beaf5f34511d449ea695732e19d1f7744d4efb21f9a9e'
    sys.path.insert(0,str(root)); from model import Brain,Config
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(4); saved=torch.load(root/'base.pt',map_location='cpu',weights_only=True)
    tok=Tokenizer.from_file(str(root/'tokenizer.json')); eos=tok.token_to_id('<|eos|>'); pad=tok.token_to_id('<|pad|>')
    out=Path('/kaggle/working/reasoning-test'); out.mkdir(exist_ok=True)
    report={'status':'running','gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'training_examples':len(DATA['train']),'replay_examples':len(DATA['replay']),'test_examples':len(DATA['test']),'epochs':3,'learning_rate':1e-5,'global_batch':16,'seed':21911,'api_calls':0,'automatic_promotion':False,'conditions':[],'arms':[],'limitations':'Small synthetic diagnostic. Same examples and optimizer step limit, not equal FLOPs or supervised tokens. Exact final vocabulary; comparison check requires prescribed numerical fact.'}
    def log(event):
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(event,ensure_ascii=False),flush=True)
    def fresh():
        torch.manual_seed(21911); model=Brain(Config(**saved['config'])); model.load_state_dict(saved['model']); return model.cuda()
    def encoded(rows):
        result=[]
        for p,a in rows:
            pi=tok.encode(p).ids; ai=tok.encode(a+'<|eos|>').ids
            assert pi and ai[-1]==eos and len(pi+ai)<384
            result.append((pi+ai[:-1],[-100]*(len(pi)-1)+ai))
        return result
    def batch(rows):
        length=max(len(x) for x,y in rows); x=torch.full((len(rows),length),pad,dtype=torch.long); y=torch.full_like(x,-100)
        for i,(a,b) in enumerate(rows): x[i,:len(a)]=torch.tensor(a); y[i,:len(b)]=torch.tensor(b)
        return x.cuda(),y.cuda()
    def test(model,label,mode):
        model.eval(); results=[]; pairs={}
        with torch.no_grad():
            for i,r in enumerate(DATA['test']):
                x=torch.tensor([tok.encode(prompt(r,mode)).ids],device='cuda'); ids=[]
                for _ in range(64 if mode=='reason' else 24):
                    with torch.autocast('cuda',dtype=torch.float16): token=model(x)[:,-1].argmax(-1,keepdim=True)
                    if token.item()==eos: break
                    ids.append(token.item()); x=torch.cat((x,token),1)
                text=tok.decode(ids); correct,check=score(text,r,mode)
                results.append({**r,'output':text,'correct':correct,'comparison_correct':check})
                pairs.setdefault(r['pair_id'],[]).append(correct)
                if (i+1)%25==0: log({'event':'evaluation','condition':label,'mode':mode,'done':i+1})
        condition={'label':label,'mode':mode,'correct':sum(r['correct'] for r in results),'total':len(results),'correct_pairs':sum(len(v)==2 and all(v) for v in pairs.values()),'total_pairs':len(pairs),'valid_comparison_and_answer':sum(r['correct'] and r['comparison_correct'] is True for r in results) if mode=='reason' else None,'by_family':{family:{'correct':sum(r['correct'] for r in results if r['family']==family),'total':sum(r['family']==family for r in results)} for family in ['loss','budget','promotion','gpu','addition']},'by_phrasing':{group:{'correct':sum(r['correct'] for r in results if r['phrasing']==group),'total':sum(r['phrasing']==group for r in results)} for group in ['familiar','new']},'results':results}
        report['conditions'].append(condition); log({k:v for k,v in condition.items() if k!='results'})
    base=fresh(); test(base,'base','direct'); test(base,'base','reason'); del base; torch.cuda.empty_cache()
    class WeightedLoss(torch.nn.Module):
        def __init__(self,net): super().__init__(); self.net=net
        def forward(self,x,y):
            with torch.autocast('cuda',dtype=torch.float16): loss=self.net(x,y)
            count=(y!=-100).sum(); return torch.stack((loss*count,count))
    for mode in ['direct','reason']:
        model=fresh(); wrapped=WeightedLoss(model)
        parallel=torch.nn.DataParallel(wrapped) if torch.cuda.device_count()>1 else wrapped
        def loss_for(x,y):
            values=parallel(x,y).reshape(-1,2); return values[:,0].sum()/values[:,1].sum()
        train=encoded([(prompt(r,mode),target(r,mode)) for r in DATA['train']]+[(r['prompt'],r['response']) for r in DATA['replay']])
        dev=encoded([(prompt(r,mode),target(r,mode)) for r in DATA['dev']])
        @torch.no_grad()
        def dev_loss():
            model.eval(); total=0.; count=0
            for cursor in range(0,len(dev),16):
                x,y=batch(dev[cursor:cursor+16]); n=int((y!=-100).sum()); value=loss_for(x,y)
                if not torch.isfinite(value): raise RuntimeError('Nonfinite dev loss')
                total+=value.item()*n; count+=n
            model.train(); return total/count
        arm={'mode':mode,'epochs':[],'steps':0,'supervised_tokens':0,'initial_loss':dev_loss()}
        report['arms'].append(arm); best=arm['initial_loss']; checkpoint=out/(mode+'-best.pt')
        def save():
            tmp=checkpoint.with_suffix('.tmp')
            torch.save({'config':asdict(model.cfg),'model':{k:v.detach().cpu().half() for k,v in model.state_dict().items()},'validation_loss':best,'steps':arm['steps'],'mode':mode,'base_checkpoint_sha256':hashlib.sha256((root/'base.pt').read_bytes()).hexdigest()},tmp); tmp.replace(checkpoint)
        save()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024); rng=random.Random(21911); start=time.monotonic(); overflow=0
        for epoch in range(3):
            order=list(range(len(train))); rng.shuffle(order)
            for cursor in range(0,len(order),16):
                if time.monotonic()-start>600: break
                x,y=batch([train[i] for i in order[cursor:cursor+16]])
                opt.zero_grad(set_to_none=True); loss=loss_for(x,y)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss).backward(); scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):
                    scaler.update(new_scale=scaler.get_scale()/2); overflow+=1
                    if overflow>16: raise RuntimeError('Repeated gradient overflow')
                    continue
                scaler.step(opt); scaler.update(); overflow=0
                arm['steps']+=1; arm['supervised_tokens']+=int((y!=-100).sum())
            current=dev_loss(); arm['epochs'].append({'epoch':epoch+1,'dev_loss':current})
            if current<best: best=current; save()
            log({'event':'training_epoch','arm':mode,'epoch':epoch+1,'dev_loss':current,'steps':arm['steps']})
            if time.monotonic()-start>600: break
        arm.update(best_dev_loss=best,training_seconds=round(time.monotonic()-start),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        selected=torch.load(checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(selected['model']); del selected
        test(model,'trained-'+mode,mode)
        del model,parallel,wrapped,opt,scaler; torch.cuda.empty_cache()
    report.update(status='completed_candidate_only')
    log({'event':'completed','conditions':[{k:r[k] for k in ['label','mode','correct','total','correct_pairs','valid_comparison_and_answer']} for r in report['conditions']]})
if __name__=='__main__': main()
