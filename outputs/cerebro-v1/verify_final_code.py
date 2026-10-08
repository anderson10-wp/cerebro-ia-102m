import ast
import json
from pathlib import Path

folder = Path(__file__).parent / 'test-final'
data = json.loads((folder / 'generation-results.json').read_text(encoding='utf-8'))
item = data['results'][-1]
code = item['prompt'] + item['continuation']
result = {'python_code': code, 'syntax_valid': False,
          'correct_for_numeric_addition': False,
          'note': 'Generated a.sum(2); numeric inputs have no sum method. Generated code was not executed.'}
try:
    ast.parse(code)
    result['syntax_valid'] = True
except SyntaxError as error:
    result['syntax_error'] = str(error)
(folder / 'code-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=True))
