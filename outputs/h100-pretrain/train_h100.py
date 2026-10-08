"""Resume our 102M checkpoint on one H100; optimize batch throughput, preserve data identity."""
import argparse, hashlib, json, math, os, signal, time, subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from model import Brain, Config
from training_helpers import atomic_save, affine_order, packed_batch, cpu_weights, DOMAINS, SCHEDULE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'result'
stopping=False
def stop_signal(*_):
    global stopping
    stopping=True

def file_hash(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

@torch.no_grad()
def evaluate(model,arrays):
    model.eval(); values={}
    for domain in DOMAINS:
        rng=np.random.default_rng(7297); losses=[]
        n=(len(arrays[domain])-1)//model.cfg.context
        for _ in range(8):
            x,y=packed_batch(arrays[domain],rng.integers(0,n,size=4),model.cfg.context,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(x,y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite validation loss')
            losses.append(loss.item())
        values[domain]=sum(losses)/len(losses)
    values['weighted']=.6*values['pt']+.25*values['en']+.15*values['code']
    model.train(); return values

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seconds',type=int,default=1200)
    args=parser.parse_args()
    if not 60<=args.seconds<=16200: raise ValueError('Bounded first run only')
    started=time.monotonic()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name():
        raise RuntimeError('Exactly one H100 is required')
    signal.signal(signal.SIGTERM,stop_signal); signal.signal(signal.SIGINT,stop_signal)
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(20261006)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
    data=ROOT/'data'; manifest=json.loads((data/'manifest.json').read_text())
    corpus_hash=file_hash(data/'manifest.json'); tokenizer_hash=file_hash(data/'tokenizer.json')
    saved=torch.load(ROOT/'source.pt',map_location='cpu',weights_only=False)
    cfg=Config(**saved['config'])
    if corpus_hash!=saved['corpus_hash'] or tokenizer_hash!=saved['tokenizer_hash']:
        raise RuntimeError('Corpus or tokenizer differs from original checkpoint')
    arrays={split:{} for split in ('train','validation')}
    for split in arrays:
        for domain in DOMAINS:
            path=data/f'{domain}-{split}.bin'
            if path.stat().st_size!=manifest['counts'][domain][split]*2 or file_hash(path)!=manifest['sha256'][path.name]:
                raise RuntimeError('Corpus size or hash mismatch: '+path.name)
            arrays[split][domain]=np.memmap(path,dtype=np.uint16,mode='r')
    model=Brain(cfg).cuda(); model.load_state_dict(saved['model'])
    groups=[{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.1},
            {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}]
    optimizer=torch.optim.AdamW(groups,lr=3e-5,betas=(.9,.95),fused=True)
    optimizer.load_state_dict(saved['optimizer'])
    tokens=int(saved['tokens']); initial_tokens=tokens; step=int(saved['step']); counts=dict(saved['counts'])
    del saved
    report={'status':'benchmarking','parameters':model.parameter_count(),'source_tokens':initial_tokens,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'precision':'BF16 autocast; FP32 master weights and AdamW state',
            'optimizer_resumed':True,'global_batch_tokens':65536,'benchmark':[],
            'corpus_hash':corpus_hash,'tokenizer_hash':tokenizer_hash,'dataset_note':'Repeated passes through the existing 2B-token training corpus; not additional unique tokens.',
            'original_model_replaced':False,'audit_used_for_selection':False}
    def log(value):
        value={'step':step,'tokens':tokens,**value}
        with (OUT/'metrics.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    def persist_report():
        (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    persist_report()
    # Backward-only benchmark: no optimizer updates, no changed data cursors,
    # no benchmark tokens counted as training. All choices keep 64 sequences/update.
    for batch in (8,16,32,64):
        if stopping: break
        x=y=loss=None
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            x,y=packed_batch(arrays['train']['pt'],np.arange(batch),cfg.context,'cuda')
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(x,y)
                loss.backward()
            torch.cuda.synchronize(); tick=time.monotonic()
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(x,y)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite benchmark loss')
                loss.backward()
            torch.cuda.synchronize()
            report['benchmark'].append({'micro_batch':batch,'tokens_per_second':4*batch*cfg.context/(time.monotonic()-tick),
                                        'peak_memory_bytes':torch.cuda.max_memory_allocated(),'status':'ok'})
        except torch.cuda.OutOfMemoryError:
            report['benchmark'].append({'micro_batch':batch,'status':'out_of_memory'})
        finally:
            optimizer.zero_grad(set_to_none=True)
            x=y=loss=None
            torch.cuda.empty_cache()
        persist_report()
    valid=[b for b in report['benchmark'] if b['status']=='ok' and b['peak_memory_bytes']<torch.cuda.get_device_properties(0).total_memory*.90]
    if not valid: raise RuntimeError('No safe H100 batch configuration')
    batch=max(valid,key=lambda b:b['tokens_per_second'])['micro_batch']; accumulation=64//batch
    baseline=evaluate(model,arrays['validation']); best=baseline['weighted']
    report.update(status='training',micro_batch=batch,accumulation=accumulation,baseline_validation=baseline)
    atomic_save({'config':asdict(cfg),'model':cpu_weights(model),'tokens':tokens,'validation':baseline},OUT/'best-model.pt')
    (OUT/'tokenizer.json').write_bytes((data/'tokenizer.json').read_bytes())
    sizes={d:(len(arrays['train'][d])-1)//cfg.context for d in DOMAINS}
    orders={d:affine_order(sizes[d],3000+i) for i,d in enumerate(DOMAINS)}
    train_start=time.monotonic(); deadline=min(train_start+args.seconds,started+16500); last_save=train_start
    log({'event':'baseline','validation':baseline,'micro_batch':batch,'accumulation':accumulation})
    bad_checks=0
    def checkpoint(event):
        nonlocal best,last_save,bad_checks
        values=evaluate(model,arrays['validation'])
        if values['weighted']<best:
            best=values['weighted']
            atomic_save({'config':asdict(cfg),'model':cpu_weights(model),'tokens':tokens,'validation':values},OUT/'best-model.pt')
        if event=='validation':
            bad_checks=bad_checks+1 if values['weighted']>baseline['weighted']+.01 else 0
        atomic_save({'config':asdict(cfg),'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                     'tokens':tokens,'counts':counts,'best_validation':best,'corpus_hash':corpus_hash,'tokenizer_hash':tokenizer_hash,
                     'precision':'bf16','batch_geometry':[1,batch,accumulation]},OUT/'last.pt')
        atomic_save({'config':asdict(cfg),'model':cpu_weights(model),'tokens':tokens},OUT/'model.pt')
        report.update(tokens_processed=tokens,tokens_this_run=tokens-initial_tokens,steps=step,best_validation_loss=best,
                      latest_validation=values,elapsed_seconds=time.monotonic()-started)
        log({'event':event,'validation':values}); persist_report(); last_save=time.monotonic()
    checkpoint('initial_checkpoint')
    window=time.monotonic(); window_tokens=tokens
    while not stopping and bad_checks<2 and time.monotonic()<deadline and tokens-initial_tokens<100_000_000_000:
        domain=SCHEDULE[step%len(SCHEDULE)]; a,b=orders[domain]; n=sizes[domain]
        fraction=min(1.,(time.monotonic()-train_start)/args.seconds)
        lr=1e-5+.5*(3e-5-1e-5)*(1+math.cos(math.pi*fraction))
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True); total_loss=0.
        for micro in range(accumulation):
            indexes=((np.arange(counts[domain]+micro*batch,counts[domain]+(micro+1)*batch,dtype=np.int64)%n)*a+b)%n
            x,y=packed_batch(arrays['train'][domain],indexes,cfg.context,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(x,y)/accumulation
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward(); total_loss+=loss.detach().item()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('Non-finite gradient')
        optimizer.step(); counts[domain]+=64; step+=1; tokens+=65536
        if step%20==0:
            torch.cuda.synchronize(); now=time.monotonic()
            gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
            log({'event':'train','domain':domain,'loss':total_loss,'lr':lr,'tokens_per_second':(tokens-window_tokens)/(now-window),'gpu_utilization_memory':gpu})
            window=now; window_tokens=tokens
        if time.monotonic()-last_save>=180:
            checkpoint('validation'); window=time.monotonic(); window_tokens=tokens
    checkpoint('final_validation')
    report.update(status='stopped_safely' if stopping else 'completed',
                  stop_reason='signal' if stopping else ('validation_regressed_twice' if bad_checks>=2 else 'time_or_token_limit'),
                  validation_improved=best<baseline['weighted'],training_seconds=time.monotonic()-train_start)
    persist_report()

if __name__=='__main__':
    main()
