"""Verify recovered inputs and create an explicit credential-free upload manifest."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
manifest_path=ROOT/'data/corpus/manifest.json'
manifest=json.loads(manifest_path.read_text())
source_path=ROOT/'source/brain/last.pt'
if source_path.stat().st_size<1_000_000_000: raise RuntimeError('Original optimizer checkpoint is not downloaded')
source=torch.load(source_path,map_location='cpu',weights_only=False)
if source['tokens']!=1_999_962_112 or source['corpus_hash']!=digest(manifest_path): raise RuntimeError('Wrong source checkpoint')
if source['tokenizer_hash']!=digest(ROOT/'data/corpus/tokenizer.json'): raise RuntimeError('Wrong tokenizer')
if sum(source['counts'].values())*1024!=source['tokens']: raise RuntimeError('Source data cursors do not match token count')
if not all(torch.isfinite(v).all().item() for v in source['model'].values()): raise RuntimeError('Source weights are not finite')
del source
paths=[(name,name) for name in ('model.py','training_helpers.py','train_h100.py')]
paths+=[('data/corpus/'+name,'data/'+name) for name in ('manifest.json','tokenizer.json')]
paths+=[('source/brain/last.pt','source.pt')]
for domain in ('pt','en','code'):
    for split in ('train','validation'):
        name=domain+'-'+split+'.bin'; path=ROOT/'data/corpus'/name
        if path.stat().st_size!=manifest['counts'][domain][split]*2 or digest(path)!=manifest['sha256'][name]:
            raise RuntimeError('Incomplete or changed corpus: '+name)
        paths.append(('data/corpus/'+name,'data/'+name))
files=[{'local':local,'remote':remote,'bytes':(ROOT/local).stat().st_size,'sha256':digest(ROOT/local)} for local,remote in paths]
value={'files':files,'source_tokens':1_999_962_112,'total_bytes':sum(f['bytes'] for f in files),
       'credentials_included':False,'precision':'BF16','global_batch_tokens':65536,'budget_credits':3}
(ROOT/'bundle-manifest.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
print(json.dumps({'status':'verified','files':len(files),'bytes':value['total_bytes']}))
