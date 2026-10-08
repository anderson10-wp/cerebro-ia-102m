"""Local, unconstrained greedy evaluation. No training or tool execution."""
import argparse,hashlib,json,sys,time,unicodedata
from pathlib import Path
import torch
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'h100-pretrain'))
from model import Brain,Config

def score(output,item):
    if item['scorer']=='exact':
        return unicodedata.normalize('NFC',output.strip())==unicodedata.normalize('NFC',item['response'].strip())
    try:
        # Canonical strings preserve boolean vs numeric value types.
        actual=json.dumps(json.loads(output),sort_keys=True,ensure_ascii=False,allow_nan=False)
        expected=json.dumps(json.loads(item['response']),sort_keys=True,ensure_ascii=False,allow_nan=False)
        return actual==expected
    except (TypeError,ValueError): return False

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=ROOT.parent/'h100-pretrain/result/best-model.pt')
    p.add_argument('--split',choices=['validation','audit'],default='validation')
    p.add_argument('--out',type=Path,default=ROOT/'baseline.json')
    p.add_argument('--cases',type=Path)
    p.add_argument('--device',choices=['cpu','directml'],default='cpu')
    args=p.parse_args()
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    path=ROOT/(args.split+'.jsonl')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[args.split]['sha256']
    if args.cases: path=args.cases
    tokenizer=ROOT.parent/'h100-pretrain/result/tokenizer.json'
    assert hashlib.sha256(tokenizer.read_bytes()).hexdigest()==manifest['tokenizer_sha256']
    cases=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert all(score(c['response'],c) for c in cases)
    assert not score('{"ativo":0}',{'scorer':'json','response':'{"ativo":false}'})
    torch.set_num_threads(2); torch.manual_seed(42)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model']); net.eval()
    device=torch.device('cpu'); hardware='cpu'
    if args.device=='directml':
        from directml_backend import prepare_directml
        device,hardware=prepare_directml(net)
    tok=Tokenizer.from_file(str(tokenizer)); eos=tok.token_to_id('<|eos|>')
    report={'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'split':args.split,'decode':'greedy, maximum 48 new tokens, stop at EOS; no output repair or grammar constraints',
        'weights_updated':False,'tools_executed':0,'results':[], 'hardware':hardware,
        'cases_path':str(path),'cases_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with torch.no_grad():
        for case in cases:
            x=torch.tensor([tok.encode(case['prompt']).ids],device=device); generated=[]; tick=time.monotonic()
            for _ in range(48):
                value=net(x)[:,-1].argmax(-1,keepdim=True)
                if value.item()==eos: break
                generated.append(value.item()); x=torch.cat([x,value],1)
            output=tok.decode(generated)
            item={'id':case['id'],'category':case['category'],'prompt':case['prompt'],'expected':case['response'],
                'output':output,'correct':score(output,case),'seconds':round(time.monotonic()-tick,2)}
            report['results'].append(item)
            report['correct']=sum(r['correct'] for r in report['results']); report['evaluated']=len(report['results'])
            args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'id':case['id'],'correct':item['correct'],'done':report['evaluated']}),flush=True)
    print(json.dumps({'correct':report['correct'],'total':len(cases),'split':args.split}))
if __name__=='__main__': main()
