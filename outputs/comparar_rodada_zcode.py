"""Compara questao a questao a v5 (antes) com a rodada local nova (depois)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
before = json.loads((ROOT / 'teacher-v5/candidate-validation.json').read_text(encoding='utf-8'))
after = json.loads((ROOT / 'local-run-20261007-182714-d01735/evaluation.json').read_text(encoding='utf-8'))
old = {r['id']: r for r in before['results']}

for r in after['results']:
    o = old[r['id']]
    if o['correct'] != r['correct']:
        mark = 'MUDOU: ' + ('erro->acerto' if r['correct'] else 'ACERTO->erro')
    else:
        mark = 'acerto mantido' if r['correct'] else 'erro mantigo'
    out = r['output'][:70].replace('\n', ' ')
    print(f"{r['id']} [{mark}]")
    if o['correct'] != r['correct']:
        print(f"   antes : {o['output'][:70]}")
        print(f"   agora : {out}")
    print(f"   agora : {out}")
