"""Mechanical checks; preserve qualitative review as a separate decision."""
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'agent-pilot'))
from routing import validate_route

def main():
    report=json.loads((ROOT/'result/report.json').read_text(encoding='utf-8'))
    cases={c['id']:c for c in json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))}
    scored=[]
    for result in report['results']:
        case=cases[result['id']]
        verdict={'id':result['id'],'model':result['model'],'finish_reason':result['finish_reason'],
                 'passed':None,'semantic_review_needed':False}
        if case['kind']=='native_tool':
            calls=result.get('tool_calls') or []
            try:
                if len(calls)!=1:
                    raise ValueError('Expected exactly one native tool call')
                route={'tool':calls[0]['function']['name'],'arguments':json.loads(calls[0]['function']['arguments'])}
                validate_route(route,case)
                correct=route['tool']==case['expected']['tool']
                if correct and route['tool']=='calculator':
                    correct=''.join(route['arguments']['expression'].split())==''.join(case['expected']['arguments']['expression'].split())
                if correct and route['tool']=='teacher':
                    correct=route['arguments']==case['expected']['arguments']
                verdict.update({'route':route,'format_valid':True,'passed':correct,
                                'semantic_review_needed':route['tool']=='clarify'})
            except (ValueError,TypeError,KeyError) as e:
                verdict.update({'format_valid':False,'passed':False,'error':str(e)})
        elif case['kind']=='arithmetic':
            verdict['passed']=(result.get('answer') or '').strip()=='372'
        elif case['kind']=='tool':
            try:
                value=json.loads(result.get('answer') or '')
                valid=set(value)=={'tool','arguments','success_criteria'} and isinstance(value['arguments'],dict) and set(value['arguments'])=={'task','code'} and isinstance(value['success_criteria'],list)
                verdict.update({'format_valid':valid,'passed':valid and value['tool']=='python_reviewer','semantic_review_needed':True})
            except (ValueError,TypeError,KeyError) as e:
                verdict.update({'format_valid':False,'passed':False,'error':str(e)})
        else:
            verdict['semantic_review_needed']=True
        if result['finish_reason']=='length':
            verdict['truncated']=True
            verdict['passed']=False
        scored.append(verdict)
    summary={}
    for model in ('gpt-oss-20b','gpt-oss-120b'):
        subset=[s for s in scored if s['model']==model]
        native=[s for s in subset if cases[s['id']]['kind']=='native_tool']
        times=[r['seconds'] for r in report['results'] if r['model']==model]
        summary[model]={'completed_cases':len(subset),'native_mechanical_correct':sum(s['passed'] is True for s in native),
                        'native_cases':len(native),'native_semantic_reviews_needed':sum(s['semantic_review_needed'] for s in native),
                        'truncated':sum(s.get('truncated',False) for s in subset),
                        'median_case_seconds':statistics.median(times) if times else None,
                        'note':'Latency includes the model-specific amount of generated reasoning; not a pure throughput measurement.'}
    output={'status':report['status'],'summary':summary,'checks':scored,
            'note':'Clarification relevance, code correctness, factual uncertainty and experiment recommendations need explicit content review. Mechanical scores alone do not approve a teacher.'}
    (ROOT/'mechanical-review.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':output['status'],'summary':summary},ensure_ascii=False))

if __name__=='__main__':
    main()
