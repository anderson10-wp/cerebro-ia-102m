"""Inference-only retention gate for the direct-answer candidate."""
import hashlib,json,sys,time,unicodedata
from pathlib import Path
import torch
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'h100-pretrain'))
from model import Brain,Config
from directml_backend import prepare_directml

def score(text,r):
    if r['kind']=='open': return None
    if r['kind']=='exact': return unicodedata.normalize('NFC',text.strip())==unicodedata.normalize('NFC',r['response'].strip())
    try: return json.dumps(json.loads(text),sort_keys=True,allow_nan=False)==json.dumps(json.loads(r['response']),sort_keys=True,allow_nan=False)
    except (ValueError,TypeError): return False
def main():
    path=ROOT/'reasoning-result/reasoning-test/direct-best.pt'
    expected='9af06f42c52b6cc974c12f7a34c934b35f6fb833245de9ee9c70a44110d8e2e1'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Checkpoint not downloaded completely or differs'
    report_path=ROOT/'reasoning-experiment/retention.json'
    if report_path.exists(): raise ValueError('Preserve existing result; do not overwrite')
    general=json.loads((ROOT/'local-run-20261007-212517-0aec1e/evaluation.json').read_text(encoding='utf-8'))
    previous=json.loads((ROOT/'kaggle-sft-review/before.json').read_text(encoding='utf-8'))
    old={('general',r['id']):r['correct'] for r in general['results']}
    old.update({('review',r['id']):r['correct'] for r in previous['results']})
    cases=[]
    for line in (ROOT/'teacher-v2/validation.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line); cases.append({**r,'kind':r['scorer'],'suite':'general','max_tokens':48})
    cases += [{**r,'suite':'review','max_tokens':64 if r['kind']=='open' else 24} for r in [json.loads(line) for line in (ROOT/'kaggle-sft-review/cases.jsonl').read_text(encoding='utf-8').splitlines()]]
    saved=torch.load(path,map_location='cpu',weights_only=True); net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model']); del saved
    torch.set_num_threads(2); torch.manual_seed(42); device,hardware=prepare_directml(net); net.eval()
    tok=Tokenizer.from_file(str(ROOT/'h100-pretrain/result/tokenizer.json')); eos=tok.token_to_id('<|eos|>')
    report={'status':'running','hardware':hardware,'checkpoint_sha256':expected,'weights_updated':False,'api_teacher_calls':0,'automatic_promotion':False,'results':[]}
    def write(): report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with torch.no_grad():
        for r in cases:
            x=torch.tensor([tok.encode(r['prompt']).ids],device=device); ids=[]
            for _ in range(r['max_tokens']):
                token=net(x)[:,-1].argmax(-1,keepdim=True)
                if token.item()==eos: break
                ids.append(token.item()); x=torch.cat((x,token),1)
            text=tok.decode(ids); correct=score(text,r)
            report['results'].append({**r,'output':text,'correct':correct,'previous_correct':old[(r['suite'],r['id'])]})
            write(); print(json.dumps({'done':len(report['results']),'suite':r['suite'],'correct':correct}),flush=True)
    report['suites']={s:{'previous_correct':sum(r['previous_correct'] is True for r in report['results'] if r['suite']==s),'correct':sum(r['correct'] is True for r in report['results'] if r['suite']==s),'total':sum(r['kind']!='open' for r in report['results'] if r['suite']==s),'regressions':[r['id'] for r in report['results'] if r['suite']==s and r['previous_correct'] is True and r['correct'] is False]} for s in ['general','review']}
    report.update(status='complete',retention_passed=all(not s['regressions'] for s in report['suites'].values()))
    write(); print(json.dumps({'suites':report['suites'],'retention_passed':report['retention_passed']}),flush=True)
if __name__=='__main__': main()
