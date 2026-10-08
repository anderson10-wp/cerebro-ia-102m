"""Bounded in-memory GPU test; never write or promote updated model weights."""
import argparse,json,statistics,sys,time,traceback,warnings
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Brain,Config
from directml_backend import prepare_directml
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--device',choices=['cpu','directml'],required=True)
    parser.add_argument('--sft',action='store_true',help='Use actual masked examples and gradient clipping')
    args=parser.parse_args(); torch.set_num_threads(4); torch.manual_seed(42)
    report={'device_requested':args.device,'torch':torch.__version__,'weights_saved':False,'cloud_used':False}
    dest=ROOT/('benchmark-'+args.device+('-sft' if args.sft else '')+'.json')
    caught=[]
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            saved=torch.load(ROOT/'result/best-model.pt',map_location='cpu',weights_only=True)
            net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model']); del saved
            device=torch.device('cpu')
            if args.device=='directml': device,report['adapter']=prepare_directml(net)
            report['parameter_device']=str(next(net.parameters()).device)
            tok=Tokenizer.from_file(str(ROOT/'result/tokenizer.json'))
            ids=torch.tensor([tok.encode('A inteligência artificial pode ajudar as pessoas a').ids],device=device)
            net.eval()
            with torch.no_grad():
                probe=net(ids)[:,-1].cpu()
                report['logit_probe']=probe[0,:32].tolist()
                report['top_token']=int(probe.argmax(-1).item())
                durations=[]
                for _ in range(3):
                    x=ids.clone(); tick=time.perf_counter()
                    for step in range(16):
                        token=net(x)[:,-1].argmax(-1,keepdim=True)
                        # Synchronize each token, matching interactive chat sampling overhead.
                        token=token.cpu().to(device); x=torch.cat([x,token],1)
                    x.cpu(); durations.append(time.perf_counter()-tick)
                report['generation_16_token_seconds']=durations
                report['generation_tokens_per_second']=16/statistics.median(durations)
            print(json.dumps({'event':'inference_done','device':args.device,'tokens_per_second':report['generation_tokens_per_second']}),flush=True)
            net.train(); torch.manual_seed(42)
            x=torch.randint(7,net.cfg.vocab_size,(4,64)).to(device)
            y=torch.randint(7,net.cfg.vocab_size,(4,64)).to(device)
            if args.sft:
                rows=[json.loads(line) for line in (ROOT.parent/'teacher-v3/train.jsonl').read_text(encoding='utf-8').splitlines()]
                examples=[]
                for i in (0,20,100,144):
                    p=tok.encode(rows[i]['prompt']).ids; a=tok.encode(rows[i]['response']+'<|eos|>').ids
                    full=p+a; labels=full[1:]; labels[:len(p)-1]=[-100]*(len(p)-1)
                    examples.append((full[:-1],labels))
                length=max(len(e[0]) for e in examples)
                x=torch.full((4,length),tok.token_to_id('<|pad|>'),dtype=torch.long)
                y=torch.full((4,length),-100,dtype=torch.long)
                for i,(inputs,labels) in enumerate(examples):
                    x[i,:len(inputs)]=torch.tensor(inputs); y[i,:len(labels)]=torch.tensor(labels)
                x,y=x.to(device),y.to(device)
                report['loss_policy']='Actual teacher-v3 response-only labels; gradient clip 1.0; AdamW beta2 0.95'
            net.zero_grad(set_to_none=True)
            loss=net(x,y); loss.backward()
            report['loss_probe']=loss.detach().cpu().item()
            report['gradient_probe']=net.blocks[0].attn.qkv.weight.grad.flatten()[:32].detach().cpu().tolist()
            opt=torch.optim.AdamW(net.parameters(),lr=1e-5,betas=(.9,.95 if args.sft else .999),foreach=False)
            times=[]
            for i in range(4):
                tick=time.perf_counter(); opt.zero_grad(set_to_none=True)
                loss=net(x,y); loss.backward()
                if args.sft: torch.nn.utils.clip_grad_norm_(net.parameters(),1.,foreach=False)
                opt.step()
                loss_value=loss.detach().cpu().item()
                # Read a parameter after the optimizer step to fence queued GPU updates.
                net.blocks[0].attn.qkv.weight.flatten()[0].detach().cpu().item()
                if i: times.append(time.perf_counter()-tick)
            report.update(status='completed',training_step_seconds=times,
                median_training_step_seconds=statistics.median(times),final_loss=loss_value,
                batch_size=4,sequence_length=x.shape[1],optimizer_steps_in_memory=4)
            caught=[str(w.message) for w in captured]
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        traceback.print_exc()
    finally:
        report['warnings']=list(dict.fromkeys(caught))
        dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('logit_probe','gradient_probe')},ensure_ascii=True),flush=True)
    if report['status']!='completed': sys.exit(1)
if __name__=='__main__': main()
