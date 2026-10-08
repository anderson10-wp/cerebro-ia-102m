"""Stream private Kaggle artifacts; signed URLs remain only in process memory."""
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, os, re, sys, threading, time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

ROOT=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')

def download(url,path,expected_size=None,expected_hash=None):
    if path.exists() and expected_size and path.stat().st_size==expected_size:
        with path.open('rb') as f: found=hashlib.file_digest(f,'sha256').hexdigest()
        if expected_hash==found:
            print(json.dumps({'cached_verified':path.name}),flush=True); return
    with requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=(20,60)) as response:
        match=re.fullmatch(r'bytes 0-0/(\d+)',response.headers.get('Content-Range',''))
        if response.status_code!=206 or not match: raise RuntimeError('Artifact server did not support a verified range request')
        size=int(match[1])
    if expected_size and size!=expected_size: raise RuntimeError('Artifact size differs from our manifest')
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('wb') as f: f.truncate(size)
    lock=threading.Lock(); progress=0; started=time.monotonic()
    def part(offset):
        nonlocal progress
        end=min(size-1,offset+32*1024**2-1)
        with requests.get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(20,60)) as response:
            if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{end}/{size}':
                raise RuntimeError('Unexpected artifact range')
            received=0
            with partial.open('r+b') as f:
                f.seek(offset)
                for chunk in response.iter_content(1024**2):
                    if time.monotonic()-started>600: raise TimeoutError('Artifact download deadline exceeded')
                    f.write(chunk); received+=len(chunk)
            if received!=end-offset+1: raise RuntimeError('Incomplete artifact range')
            with lock:
                progress+=received
                print(json.dumps({'file':path.name,'downloaded_bytes':progress,'total_bytes':size}),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(part,range(0,size,32*1024**2)): pass
    with partial.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    if expected_hash and digest!=expected_hash: raise RuntimeError('Artifact hash mismatch')
    partial.replace(path)
    print(json.dumps({'verified_file':path.name,'bytes':size,'sha256':digest}),flush=True)

def main():
    api=KaggleApi(); api.authenticate()
    manifest=json.loads((ROOT/'data/corpus/manifest.json').read_text())
    plans=[('cerebro-v1-102m-corpus',ROOT/'data',set(d+'-'+s+'.bin' for d in ('pt','en','code') for s in ('train','validation'))),
           ('cerebro-v1-102m-train-03',ROOT/'source',{'last.pt'})]
    for slug,dest,names in plans:
        request=ApiListKernelSessionOutputRequest()
        request.user_name='andersonluan10ad'; request.kernel_slug=slug; request.page_size=200
        with api.build_kaggle_client() as client:
            result=client.kernels.kernels_api_client.list_kernel_session_output(request)
        seen=set()
        for item in result.files:
            name=Path(item.file_name).name
            if name not in names: continue
            path=(dest/item.file_name).resolve()
            if dest.resolve() not in path.parents: raise RuntimeError('Artifact path escaped destination')
            expected_size=None; expected_hash=None
            if name.endswith('.bin'):
                domain,split=name[:-4].split('-')
                expected_size=manifest['counts'][domain][split]*2
                expected_hash=manifest['sha256'][name]
            download(item.url,path,expected_size,expected_hash)
            seen.add(name)
        if seen!=names: raise RuntimeError('Required Kaggle artifacts were missing')

if __name__=='__main__':
    try: main()
    except Exception as error:
        # Requests exceptions contain signed URLs; report only a safe error class.
        import traceback
        frame=traceback.extract_tb(error.__traceback__)[-1]
        print(json.dumps({'status':'download_failed','error_type':type(error).__name__,'function':frame.name,'line':frame.lineno}),flush=True)
        sys.exit(1)
