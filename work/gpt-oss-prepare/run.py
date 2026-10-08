import concurrent.futures
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path('/kaggle/working/runtime')
ROOT.mkdir(parents=True, exist_ok=True)
REVISION = 'ef9b12f2ff56c69cf32153a02784e7a3c88bf524'
TAG = 'b11391'

def fetch(name, url):
    target = ROOT / name
    print(json.dumps({'event': 'download_start', 'file': name}), flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, target.open('wb') as destination:
        while chunk := source.read(8 * 1024 * 1024):
            destination.write(chunk)
    print(json.dumps({'event': 'download_done', 'file': name, 'bytes': target.stat().st_size}), flush=True)
    if name.endswith('.tar.gz'):
        with tarfile.open(target) as archive:
            archive.extractall(ROOT / 'llama', filter='data')
        target.unlink()
    else:
        target.chmod(0o755 if name == 'cloudflared' else 0o644)

jobs = [
 ('gpt-oss-20b-MXFP4.gguf', f'https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/{REVISION}/gpt-oss-20b-MXFP4.gguf'),
 ('llama.tar.gz', f'https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz'),
 ('cudart.tar.gz', f'https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz'),
 ('cloudflared', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'),
]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(fetch, *job) for job in jobs]
    for future in futures:
        future.result()
manifest = {'model_repo': 'ggml-org/gpt-oss-20b-GGUF', 'revision': REVISION,
            'llama_release': TAG, 'files': {}}
for file in ROOT.rglob('*'):
    if file.is_file():
        digest = hashlib.sha256()
        with file.open('rb') as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        manifest['files'][str(file.relative_to(ROOT))] = {'bytes': file.stat().st_size, 'sha256': digest.hexdigest()}
(ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'event': 'preparation_complete', 'manifest': manifest}), flush=True)
