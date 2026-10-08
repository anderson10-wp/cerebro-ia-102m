"""Execute only the two reviewed, simple functions, excluding generated examples."""
import ast
import json
from pathlib import Path

root = Path(__file__).parent
report = json.loads((root / 'test-results.json').read_text(encoding='utf-8'))
allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.If, ast.UnaryOp,
           ast.Not, ast.Name, ast.Load, ast.Store, ast.Raise, ast.Call, ast.Constant,
           ast.Assign, ast.For, ast.AugAssign, ast.Add, ast.Return, ast.BinOp,
           ast.Div, ast.List, ast.Expr, ast.Attribute, ast.Mult)
for item in report['results']:
    if item['test'] in ('python', 'bug'):
        tree = ast.parse(item['answer'])
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert len(tree.body) == 1
        for node in ast.walk(tree):
            assert isinstance(node, allowed), type(node)
            if isinstance(node, ast.Attribute):
                assert node.attr == 'append'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id in ('len', 'ValueError')
        scope = {'__builtins__': {'len': len, 'ValueError': ValueError}}
        exec(compile(tree, '<reviewed-function>', 'exec'), scope)
        if item['test'] == 'python':
            f = scope['media']
            checks = [f([1, 2, 3]) == 2, f([-3, 1]) == -1, f([2.5]) == 2.5]
            try:
                f([])
                checks.append(False)
            except ValueError:
                checks.append(True)
        else:
            f = scope['quadrados']
            checks = [f([1, 2, 3]) == [1, 4, 9], f([]) == [], f([-2, 0]) == [4, 0]]
        item['reviewed_verification'] = {'passed': all(checks), 'checks': checks,
            'method': 'Reviewed function only; restricted builtins; generated examples excluded'}
    elif item['test'] == 'tool':
        item['reviewed_verification'] = {'correct_tool_choice': True,
            'strict_json_format_passed': False, 'note': 'Returned Markdown code fences despite request for JSON only'}
    elif item['test'] == 'uncertainty':
        item['reviewed_verification'] = {'passed': True, 'method': 'Human review: correctly declined to infer progress without metrics'}
    elif item['test'] == 'evaluation':
        item['reviewed_verification'] = {'passed': None, 'note': 'Useful general evaluation plan, but unsubstantiated minimum of 10000 examples and rigid requirement to improve every metric; exceeds requested 150 words', 'words': len(item['answer'].split())}
(root / 'test-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
for item in report['results']:
    print(json.dumps({'test': item['test'], 'review': item.get('reviewed_verification', item.get('verification'))}, ensure_ascii=True))
