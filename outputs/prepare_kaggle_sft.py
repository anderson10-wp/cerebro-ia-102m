"""Package a private, reproducible SFT experiment. No API teachers or GPU work."""
import hashlib,json,random,re,shutil,urllib.request
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'work/kaggle-sft-data'
REV='c65b2bf91938480c0683ec396f4f11ca1a79782a'
SOURCE='nicholasKluge/instruct-aira-dataset-v2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def key(q): return re.sub(r'\W+',' ',q.casefold()).strip()
def dump(name,rows): (OUT/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'hf-curriculum/portuguese.parquet'
    if not cache.exists():
        tmp=cache.with_suffix('.partial')
        urllib.request.urlretrieve(f'https://huggingface.co/datasets/{SOURCE}/resolve/{REV}/data/portuguese-00000-of-00001.parquet',tmp)
        tmp.replace(cache)
    tok=Tokenizer.from_file(str(ROOT/'h100-pretrain/result/tokenizer.json'))
    tests=read(ROOT/'teacher-v2/validation.jsonl')+read(ROOT/'paired-evaluation/validation.jsonl')+read(ROOT/'hf-curriculum/technical-validation.jsonl')
    reserved=read(ROOT/'teacher-v2/audit.jsonl')
    replay=read(ROOT/'local-run-20261007-212517-0aec1e/train.jsonl')+read(ROOT/'curriculum-teacher/batch-002/train.jsonl')
    seen={key(x['prompt']) for x in tests+reserved+replay}
    candidates=[]
    for i,r in enumerate(pq.read_table(cache).to_pylist()):
        turns=r['conversations']
        if len(turns)!=2 or [t['role'] for t in turns]!=['user','assistant']: continue
        q,a=[t['content'].strip() for t in turns]
        p=f'Pergunta: {q}\nResposta:'
        if not q or not a or key(p) in seen: continue
        if any(s in (q+a).casefold() for s in ['<|','http://','https://','como modelo de linguagem','as an ai','normalmente cinco ou mais']): continue
        ids=tok.encode(p).ids+tok.encode(a+'<|eos|>').ids
        if not 24<=len(ids)<=384 or len(tok.encode(a).ids)<8: continue
        # Group near duplicate questions with changed numbers in the same split.
        group=re.sub(r'\d+','N',key(p))
        bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
        candidates.append(dict(id=f'aira-pt-{i}',prompt=p,response=a,source=SOURCE,revision=REV,source_row=i,bucket=bucket))
        seen.add(key(p))
    random.Random(8181).shuffle(candidates)
    train=[r for r in candidates if r['bucket']>=8][:12000]
    dev=[r for r in candidates if 4<=r['bucket']<8][:384]
    audit=[r for r in candidates if r['bucket']<4][:384]
    assert len(train)>3000 and len(dev)>=128 and len(audit)>=128
    train+=replay
    # Remove exact duplicates without changing the reserved splits.
    train=list({key(r['prompt']):r for r in train}.values())
    assert not {key(r['prompt']) for r in train}&{key(r['prompt']) for r in dev+audit+tests+reserved}
    dump('train.jsonl',train); dump('dev.jsonl',dev); dump('audit.jsonl',audit); dump('tests.jsonl',tests)
    base=ROOT/'local-run-20261007-212517-0aec1e/candidate/best-model.pt'
    assert sha(base)=='1ea8da9fa900f7f2ec1beaf5f34511d449ea695732e19d1f7744d4efb21f9a9e'
    for src,name in [(base,'base.pt'),(ROOT/'h100-pretrain/result/tokenizer.json','tokenizer.json'),(ROOT/'cerebro-v1/model.py','model.py')]: shutil.copyfile(src,OUT/name)
    shutil.copyfile(ROOT/'hf-curriculum/SOURCE-CARD.md',OUT/'AIRA-SOURCE.md')
    for file in (ROOT/'hf-curriculum').glob('*LICENSE*'): shutil.copyfile(file,OUT/file.name)
    meta={'source':SOURCE,'revision':REV,'raw_sha256':sha(cache),'counts':{'train':len(train),'dev':len(dev),'audit':len(audit),'tests':len(tests)},'train_tokens':sum(len(tok.encode(r['prompt']).ids)+len(tok.encode(r['response']+'<|eos|>').ids) for r in train),'response_tokens':sum(len(tok.encode(r['response']+'<|eos|>').ids) for r in train),'files':{p.name:sha(p) for p in OUT.iterdir() if p.name in ['train.jsonl','dev.jsonl','audit.jsonl','tests.jsonl','base.pt','tokenizer.json','model.py']},'quality':'Automatic format, length and duplicate filters; factual quality is NOT fully reviewed. Candidate only. Audit not used for checkpoint selection.','limits':{'epochs':2,'seconds':4500,'learning_rate':1e-5,'batch_size':8,'checkpoint_steps':250}}
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (OUT/'dataset-metadata.json').write_text(json.dumps({'title':'Cerebro 102M SFT October 08','id':'andersonluan10ad/cerebro-102m-sft-oct08','licenses':[{'name':'other'}]},indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)
if __name__=='__main__': main()
