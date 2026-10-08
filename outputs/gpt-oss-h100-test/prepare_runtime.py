"""Download pinned model files on CPU before renting a GPU."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'
RUNTIME.mkdir(exist_ok=True)
MODELS=[
    {'model':'gpt-oss-20b','repo':'ggml-org/gpt-oss-20b-GGUF','revision':'ef9b12f2ff56c69cf32153a02784e7a3c88bf524',
     'file':'gpt-oss-20b-MXFP4.gguf','bytes':12109566624,'sha256':'27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901'},
    {'model':'gpt-oss-120b','repo':'ggml-org/gpt-oss-120b-GGUF','revision':'238abdd290bb874b90a5da1b4549881b7d05c091',
     'file':'gpt-oss-120b-MXFP4.gguf','bytes':63387346208,'sha256':'582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d'}
]
TAG='b11391'

def digest(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(16*1024*1024):
            d.update(data)
    return d.hexdigest()

def fetch(name,url,expected_size=None,expected_sha=None):
    target=RUNTIME/name
    if target.exists() and expected_size and target.stat().st_size==expected_size and digest(target)==expected_sha:
        print(json.dumps({'event':'cached_verified','file':name}),flush=True)
        return
    temp=target.with_suffix(target.suffix+'.partial')
    total=0
    started=time.monotonic()
    print(json.dumps({'event':'download_start','file':name}),flush=True)
    with urllib.request.urlopen(url,timeout=120) as src,temp.open('wb') as dest:
        while data:=src.read(8*1024*1024):
            dest.write(data)
            total+=len(data)
    if expected_size and total!=expected_size:
        raise ValueError('Incomplete download: '+name)
    sha=digest(temp)
    if expected_sha and sha!=expected_sha:
        raise ValueError('SHA mismatch: '+name)
    temp.replace(target)
    print(json.dumps({'event':'download_verified','file':name,'bytes':total,'seconds':round(time.monotonic()-started,2),'sha256':sha}),flush=True)
    if name.endswith('.tar.gz'):
        with tarfile.open(target) as archive:
            archive.extractall(RUNTIME/'llama',filter='data')

def main():
    required=sum(m['bytes'] for m in MODELS if not (RUNTIME/m['file']).exists())+2*1024**3
    free=shutil.disk_usage(RUNTIME).free
    if free<required:
        raise RuntimeError(f'Need {required} bytes of free disk; found {free}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(fetch,m['file'],f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{m['file']}",m['bytes'],m['sha256']) for m in MODELS]
        for prefix in ('llama','cudart-llama'):
            name=f'{prefix}-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz'
            tasks.append(pool.submit(fetch,name,f'https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{name}'))
        for task in tasks:
            task.result()
    manifest={'models':MODELS,'llama_release':TAG,'binary_files':{str(p.relative_to(RUNTIME)): {'bytes':p.stat().st_size,'sha256':digest(p)} for p in (RUNTIME/'llama').rglob('*') if p.is_file()}}
    (RUNTIME/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    server=next((RUNTIME/'llama').rglob('llama-server'))
    server.chmod(0o755)
    print(json.dumps({'event':'runtime_prepared','models':len(MODELS),'llama_server':str(server)}),flush=True)

if __name__=='__main__':
    main()
