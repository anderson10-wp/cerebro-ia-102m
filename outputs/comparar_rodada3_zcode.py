"""Compara v5 com a rodada 3 (190808) mostrando as perguntas e respostas."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
run = ROOT / 'local-run-20261007-190808-8c59a4'
rep = json.loads((run / 'candidate' / 'report.json').read_text(encoding='utf-8'))
print('loss inicial:', round(rep['initial_validation_loss'], 4),
      '| melhor:', round(rep['best_validation_loss'], 4), '| passos:', rep['steps'])

before = json.loads((ROOT / 'teacher-v5/candidate-validation.json').read_text(encoding='utf-8'))
after = json.loads((run / 'evaluation.json').read_text(encoding='utf-8'))
old = {r['id']: r for r in before['results']}

for r in after['results']:
    o = old[r['id']]
    # Busca a pergunta no JSON (campo 'prompt', 'input' ou 'question')
    pergunta = r.get('prompt') or r.get('input') or r.get('question') or r.get('instruction') or 'Campo de pergunta não encontrado'
    
    status = 'REGRESSAO' if (o['correct'] and not r['correct']) else ('GANHO' if (not o['correct'] and r['correct']) else ('ACERTO' if r['correct'] else 'ERRO'))
    
    print(f"\n==========================================")
    print(f"ID       : {r['id']} [{status}]")
    print(f"PERGUNTA : {pergunta}")
    print(f"RESPOSTA : {r['output']}")