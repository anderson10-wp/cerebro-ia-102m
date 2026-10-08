"""Package only an explicit allowlist of public code/data and our model weights."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
relative = [
    'outputs/agent-pilot/pilot_train.py', 'outputs/agent-pilot/sft.py', 'outputs/agent-pilot/routing.py',
    'outputs/agent-pilot/pilot-train.jsonl', 'outputs/agent-pilot/pilot-data-manifest.json',
    'outputs/agent-pilot/sft-validation.jsonl', 'outputs/agent-pilot/routing-cases.json',
    'outputs/agent-pilot/data-review.json', 'outputs/cerebro-v1/model.py',
    'outputs/cerebro-v1/trained/brain/best-model.pt', 'outputs/cerebro-v1/trained/brain/tokenizer.json']
manifest = {}
with zipfile.ZipFile(ROOT/'pilot-bundle.zip', 'w', compression=zipfile.ZIP_STORED) as bundle:
    for name in relative:
        path = WORKSPACE/name
        manifest[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        bundle.write(path, arcname=name)
(ROOT/'bundle-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({'files': len(manifest), 'bytes': (ROOT/'pilot-bundle.zip').stat().st_size, 'credentials_included': False}))
