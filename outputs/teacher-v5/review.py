import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    report=json.loads((ROOT/'candidate/report.json').read_text())
    evaluation=json.loads((ROOT/'candidate-validation.json').read_text(encoding='utf-8'))
    previous=json.loads((ROOT.parent/'teacher-v3/candidate-validation.json').read_text(encoding='utf-8'))
    assert evaluation['evaluated']==previous['evaluated']==12
    old={r['id']:r for r in previous['results']}
    changes=[{'id':r['id'],'previous':old[r['id']]['correct'],'current':r['correct']} for r in evaluation['results'] if r['correct']!=old[r['id']]['correct']]
    review=dict(previous_correct=previous['correct'],current_correct=evaluation['correct'],total=12,changes=changes,promotion_gate=9,promotion_gate_passed=evaluation['correct']>=9,promoted=False,audit_evaluated=False,cloud_jobs_started=0)
    review.update(retained_correct=sum(old[r['id']]['correct'] and r['correct'] for r in evaluation['results']),new_correct=[r['id'] for r in changes if r['current']],regressions=[r['id'] for r in changes if not r['current']])
    (ROOT/'review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    lines=['# Rodada local de ensino v5','',f"Treino concluído em {report['elapsed_seconds']/60:.1f} minutos, com {report['steps']} atualizações.",f"Resultado nas 12 questões de desenvolvimento: {previous['correct']}/12 → {evaluation['correct']}/12.",'','Foram usados 136 exercícios novos e 256 exemplos de revisão. O treino começou no candidato v3, mantendo o modelo original intacto. CPU local; nenhum serviço pago acionado.','', 'O conjunto de desenvolvimento orientou os novos exercícios. Portanto, este resultado não é uma medida independente de inteligência ou de capacidade geral. O conjunto de auditoria permanece reservado. O candidato não foi promovido ao chat.','','## Respostas observadas','']
    for r in evaluation['results']:
        lines.extend([f"- {r['id']}: {'ACERTO' if r['correct'] else 'ERRO'}",f"  - Pergunta: {r['prompt'].replace(chr(10),' ')}",f"  - Esperado: {r['expected']}",f"  - Resposta: {r['output']}"])
    (ROOT/'resultado.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(review))
if __name__=='__main__': main()

