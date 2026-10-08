"""Diagnostico: lista conflitos entre v5 train.jsonl e exemplos dos professores. So leitura."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

rows = [json.loads(line) for line in (ROOT / 'teacher-v5/train.jsonl').read_text(encoding='utf-8-sig').splitlines()]
seen = {r['prompt'].strip().casefold(): r['response'] for r in rows}
print(f'v5: {len(rows)} exemplos')

conflicts = []
duplicates_ok = 0
source = {}  # key -> (provider, batch) de onde veio a primeira resposta aceita
for provider in ['groq', 'openrouter', 'gemini']:
    for batch in sorted((ROOT / f'{provider}-teacher').glob('batch-*')):
        report = batch / 'report.json'
        if not report.exists() or json.loads(report.read_text(encoding='utf-8')).get('status') != 'completed':
            continue
        for line in (batch / 'train.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = row['prompt'].strip().casefold()
            if key in seen and seen[key] != row['response']:
                conflicts.append({'provider': provider, 'batch': batch.name, 'prompt': row['prompt'],
                                  'first_source': source.get(key, 'v5'), 'first': seen[key],
                                  'second': row['response'], 'scorer': row.get('scorer')})
            elif key in seen:
                duplicates_ok += 1
                continue
            seen[key] = row['response']
            source[key] = f'{provider}/{batch.name}'

print(f'duplicados identicos (ok): {duplicates_ok}')
print(f'conflitos: {len(conflicts)}\n')
for c in conflicts:
    print('=' * 70)
    print(f"[conflito em {c['provider']}/{c['batch']}] scorer={c['scorer']}")
    print(f"PROMPT: {c['prompt']}")
    print(f"  primeiro ({c['first_source']}) -> {c['first']!r}")
    print(f"  segundo  ({c['provider']}/{c['batch']}) -> {c['second']!r}")
