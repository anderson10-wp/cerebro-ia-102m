"""Compara v5 com a rodada de 989 exemplos (185537)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
run = ROOT / 'local-run-20261007-185537-216b17'
rep = json.loads((run / 'candidate' / 'report.json').read_text(encoding='utf-8'))
print('loss inicial:', round(rep['initial_validation_loss'], 4),
      '| melhor:', round(rep['best_validation_loss'], 4), '| passos:', rep['steps'])
before = json.loads((ROOT / 'teacher-v5/candidate-validation.json').read_text(encoding='utf-8'))
after = json.loads((run / 'evaluation.json').read_text(encoding='utf-8'))
old = {r['id']: r for r in before['results']}
for r in after['results']:
    o = old[r['id']]
    marca = 'igual '
    if o['correct'] != r['correct']:
        marca = 'REGRESSAO' if o['correct'] else 'GANHO   '
        print(f"{r['id']} {marca}")
        print('   antes:', o['output'][:70].replace('\n', ' '))
        print('   agora:', r['output'][:70].replace('\n', ' '))
    else:
        print(f"{r['id']} {marca} ({'acerto' if r['correct'] else 'erro'}) -> {r['output'][:60].replace(chr(10), ' ')}")
