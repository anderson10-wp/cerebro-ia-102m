"""Private temporary inference session; no paid API calls or generated code execution."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request

API_KEY = '3L27mCKeih56GZmMHMfkFaIyJj5aALZjBFBR8lKV8d0'
START = time.monotonic()
MAX_SECONDS = 4 * 3600
roots = [p.parent for p in Path('/kaggle/input').rglob('gpt-oss-20b-MXFP4.gguf')]
if roots:
    ROOT = roots[0]
else:
    print(json.dumps({'event': 'preparing_runtime_on_gpu_session'}), flush=True)
    preparation = "import concurrent.futures\nimport hashlib\nimport json\nfrom pathlib import Path\nimport tarfile\nimport urllib.request\n\nROOT = Path('/kaggle/working/runtime')\nROOT.mkdir(parents=True, exist_ok=True)\nREVISION = 'ef9b12f2ff56c69cf32153a02784e7a3c88bf524'\nTAG = 'b11391'\n\ndef fetch(name, url):\n    target = ROOT / name\n    print(json.dumps({'event': 'download_start', 'file': name}), flush=True)\n    with urllib.request.urlopen(url, timeout=120) as source, target.open('wb') as destination:\n        while chunk := source.read(8 * 1024 * 1024):\n            destination.write(chunk)\n    print(json.dumps({'event': 'download_done', 'file': name, 'bytes': target.stat().st_size}), flush=True)\n    if name.endswith('.tar.gz'):\n        with tarfile.open(target) as archive:\n            archive.extractall(ROOT / 'llama', filter='data')\n        target.unlink()\n    else:\n        target.chmod(0o755 if name == 'cloudflared' else 0o644)\n\njobs = [\n ('gpt-oss-20b-MXFP4.gguf', f'https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/{REVISION}/gpt-oss-20b-MXFP4.gguf'),\n ('llama.tar.gz', f'https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz'),\n ('cudart.tar.gz', f'https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz'),\n ('cloudflared', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'),\n]\nwith concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:\n    futures = [pool.submit(fetch, *job) for job in jobs]\n    for future in futures:\n        future.result()\nmanifest = {'model_repo': 'ggml-org/gpt-oss-20b-GGUF', 'revision': REVISION,\n            'llama_release': TAG, 'files': {}}\nfor file in ROOT.rglob('*'):\n    if file.is_file():\n        digest = hashlib.sha256()\n        with file.open('rb') as source:\n            while chunk := source.read(8 * 1024 * 1024):\n                digest.update(chunk)\n        manifest['files'][str(file.relative_to(ROOT))] = {'bytes': file.stat().st_size, 'sha256': digest.hexdigest()}\n(ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))\nprint(json.dumps({'event': 'preparation_complete', 'manifest': manifest}), flush=True)\n"
    exec(compile(preparation, '<runtime-preparation>', 'exec'), {'__name__': '__main__'})
    ROOT = Path('/kaggle/working/runtime')
OUT = Path('/kaggle/working/service')
OUT.mkdir(exist_ok=True)
def event(name, **fields):
    data = {'event': name, **fields}
    print(json.dumps(data, ensure_ascii=True), flush=True)
    (OUT / 'status.json').write_text(json.dumps(data, indent=2))

subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
runtime = Path('/tmp/llama-runtime')
shutil.copytree(ROOT / 'llama', runtime, dirs_exist_ok=True)
llama = next(runtime.rglob('llama-server'))
llama.chmod(0o755)
environment = dict(os.environ)
environment['LD_LIBRARY_PATH'] = ':'.join(sorted({str(p.parent) for p in runtime.rglob('*.so*')})) + ':' + environment.get('LD_LIBRARY_PATH', '')
keyfile = Path('/tmp/gpt-oss-access.key')
keyfile.write_text(API_KEY)
keyfile.chmod(0o600)
logfile = (OUT / 'llama.log').open('w')
server = subprocess.Popen([str(llama), '-m', str(ROOT / 'gpt-oss-20b-MXFP4.gguf'),
    '--host', '127.0.0.1', '--port', '8081', '--alias', 'gpt-oss-20b',
    '-ngl', '99', '--split-mode', 'layer', '--tensor-split', '1,1',
    '-c', '8192', '-np', '1', '-b', '256', '-ub', '128',
    '--jinja', '--api-key-file', str(keyfile)], env=environment, stdout=logfile, stderr=subprocess.STDOUT)
tunnel = None
try:
    event('loading_model')
    ready = False
    for _ in range(180):
        if server.poll() is not None:
            raise RuntimeError('llama-server stopped: ' + (OUT / 'llama.log').read_text()[-12000:])
        try:
            request = urllib.request.Request('http://127.0.0.1:8081/health', headers={'Authorization': 'Bearer ' + API_KEY})
            with urllib.request.urlopen(request, timeout=5) as response:
                ready = response.status == 200
            if ready:
                break
        except Exception:
            pass
        time.sleep(2)
    if not ready:
        raise RuntimeError('Model health timeout')
    payload = json.dumps({'model': 'gpt-oss-20b', 'messages': [{'role':'user','content':'Responda em português: quanto é 7 vezes 8?'}],
                          'max_tokens': 1024, 'temperature': 0, 'reasoning_effort': 'low'}).encode()
    request = urllib.request.Request('http://127.0.0.1:8081/v1/chat/completions', data=payload,
        headers={'Content-Type':'application/json','Authorization':'Bearer '+API_KEY})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.load(response)
    event('smoke_test', answer=data['choices'][0]['message'].get('content'), model=data.get('model'))
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv'], check=True)
    cloud = Path('/tmp/cloudflared')
    shutil.copy2(ROOT / 'cloudflared', cloud)
    cloud.chmod(0o755)
    tunnel_log = (OUT / 'tunnel.log').open('w')
    tunnel = subprocess.Popen([str(cloud), 'tunnel', '--url', 'http://127.0.0.1:8081', '--protocol', 'http2', '--no-autoupdate'],
        stdout=tunnel_log, stderr=subprocess.STDOUT)
    public = None
    for _ in range(60):
        text = (OUT / 'tunnel.log').read_text()
        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', text)
        if match:
            public = match.group()
            break
        if tunnel.poll() is not None:
            raise RuntimeError('Tunnel failed: ' + text[-5000:])
        time.sleep(2)
    if not public:
        raise RuntimeError('No public tunnel URL')
    (OUT / 'endpoint.json').write_text(json.dumps({'base_url': public + '/v1', 'model': 'gpt-oss-20b', 'authentication': 'Bearer token stored locally by deployer'}))
    event('api_ready', base_url=public + '/v1', model='gpt-oss-20b', max_session_hours=4)
    while time.monotonic() - START < MAX_SECONDS:
        if server.poll() is not None or tunnel.poll() is not None:
            raise RuntimeError('Serving process terminated')
        time.sleep(30)
    event('session_time_limit_reached')
finally:
    for process in (tunnel, server):
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
    keyfile.unlink(missing_ok=True)
