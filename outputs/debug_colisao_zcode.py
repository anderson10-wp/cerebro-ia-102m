"""Reproduz a checagem do train_local.py para achar a colisao exata."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
held = set()
for split in ['validation', 'audit']:
    source = ROOT / f'teacher-v3/{split}.jsonl'
    held.update(json.loads(line)['prompt'].strip().casefold()
                for line in source.read_text(encoding='utf-8').splitlines())
print('held:', len(held))

seen = {}
rows = [json.loads(line) for line in (ROOT / 'teacher-v5/train.jsonl').read_text(encoding='utf-8-sig').splitlines()]
for r in rows:
    seen[r['prompt'].strip().casefold()] = r['response']

for provider in ['groq', 'openrouter', 'gemini', 'programmatic']:
    for batch in sorted((ROOT / f'{provider}-teacher').glob('batch-*')):
        f = batch / 'train.jsonl'
        if not f.exists():
            print(f'[{provider}/{batch.name}] sem train.jsonl')
            continue
        rep = batch / 'report.json'
        status = json.loads(rep.read_text(encoding='utf-8')).get('status') if rep.exists() else 'sem-report'
        print(f'[{provider}/{batch.name}] status={status}')
        if status != 'completed':
            continue
        for i, line in enumerate(f.read_text(encoding='utf-8').splitlines()):
            row = json.loads(line)
            key = row['prompt'].strip().casefold()
            if row.get('checked') is not True:
                print(f'  SEM CHECK linha {i}: {row["prompt"][:80]}')
            if key in held:
                print(f'  COLISAO linha {i}: {row["prompt"][:120]}')
            if key in seen:
                if seen[key] != row['response']:
                    print(f'  CONFLITO linha {i}: {row["prompt"][:80]}')
                else:
                    print(f'  duplicado ok linha {i}')
            seen[key] = row['response']
