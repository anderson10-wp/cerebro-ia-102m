"""Small cached Portuguese sample; original, checkable training-operation lessons."""
import hashlib,json,urllib.request,urllib.parse
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent; OUT=ROOT.parent
DATASET='nicholasKluge/instruct-aira-dataset-v2'
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    tok=Tokenizer.from_file(str(OUT/'h100-pretrain/result/tokenizer.json'))
    cache=ROOT/'sample.json'
    if not cache.exists():
        records=[]
        for offset in [0,100,200,300,400]:
            url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(dataset=DATASET,config='default',split='portuguese',offset=offset,length=100))
            with urllib.request.urlopen(url,timeout=40) as response: records.extend(json.load(response)['rows'])
        write(cache,records)
    records=json.loads(cache.read_text(encoding='utf-8'))
    held=set()
    for split in ['validation','audit']:
        held.update(json.loads(line)['prompt'].strip().casefold() for line in (OUT/f'teacher-v3/{split}.jsonl').read_text(encoding='utf-8').splitlines())
    selected=[]; seen=set(held)
    for record in records:
        turns=record['row']['conversations']
        if record.get('truncated_cells') or len(turns)!=2 or [x['role'] for x in turns]!=['user','assistant']: continue
        q,a=[x['content'].strip() for x in turns]
        prompt=f'Pergunta: {q}\nResposta:'
        if not a or prompt.casefold() in seen or '<|' in q+a: continue
        if len(tok.encode(prompt+a+'<|eos|>').ids)>180: continue
        seen.add(prompt.casefold())
        selected.append(dict(id='aira-'+str(record['row_idx']),split='train',category='portuguese_conversation',prompt=prompt,response=a,scorer='exact',source_url='https://huggingface.co/datasets/'+DATASET,source_row=record['row_idx'],license='Apache-2.0 as declared by publisher',checked=False,validation_level='format_length_only; pending content review'))
        if len(selected)==24: break
    write(ROOT/'review-candidates.json',selected)
    print(json.dumps({'downloaded_rows':len(records),'short_candidates':len(selected),'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}))
    for r in selected: print(json.dumps({'id':r['id'],'question':r['prompt'],'answer':r['response']},ensure_ascii=False))
if __name__=='__main__': main()
