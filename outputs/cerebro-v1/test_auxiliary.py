"""Probe user-provided auxiliary endpoint without executing generated commands."""
import ast
import json
import os
import time
import urllib.request
from pathlib import Path

URL = os.environ.get('GPT_OSS_ENDPOINT', 'https://grumpy-kings-greet.loca.lt/v1/chat/completions')
CASES = [
 ('arithmetic', 'Responda apenas com o número: (17 * 23) - 19.'),
 ('python', 'Escreva apenas código Python, sem markdown. Implemente media(valores), que retorna a média de uma lista de números e lança ValueError se a lista estiver vazia. Não use imports.'),
 ('bug', 'Escreva apenas código Python corrigido, sem markdown. Esta função deveria retornar o quadrado de TODOS os elementos, mas tem um bug:\ndef quadrados(xs):\n    out = []\n    for x in xs:\n        out.append(x*x)\n        return out\nCorrija também para lista vazia.'),
 ('tool', 'Você coordena outras IAs. Ferramentas disponíveis: python_reviewer (revisa código Python; não executa), web_researcher (busca fontes; não executa). Objetivo: revisar a função def soma(a,b): return a-b. Responda SOMENTE JSON com tool, arguments (objeto contendo task e code), success_criteria (lista). Escolha uma ferramenta existente; não diga que já executou.'),
 ('uncertainty', 'Você recebeu apenas este log: status=RUNNING. Nenhum contador de tokens, nenhuma métrica ou checkpoint está disponível. O usuário pergunta: o treinamento já chegou a 90% e está aprendendo bem? Responda em português, até 80 palavras, usando somente a evidência fornecida.'),
 ('evaluation', 'Temos um modelo de linguagem de 102 milhões de parâmetros treinado do zero. Ele errou 2+2 e gerou uma função soma incorreta. Um auxiliar propôs novas alterações no treinamento. Como decidir objetivamente se uma nova versão merece substituir a anterior? Responda em português, até 150 palavras. Considere orçamento limitado, testes independentes e possibilidade de regressão.'),
]

def extract_code(text):
    if '```' in text:
        text = text.split('```', 2)[1]
        if text.startswith('python'):
            text = text[6:]
    return text.strip()

def verify(name, answer):
    if name == 'arithmetic':
        return {'passed': answer.strip() == '372', 'expected': '372'}
    if name in ('python', 'bug'):
        code = extract_code(answer)
        tree = ast.parse(code)
        # This deliberately narrow harness accepts only a simple function and expressions.
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
                   ast.If, ast.UnaryOp, ast.Not, ast.Name, ast.Load, ast.Store,
                   ast.Call, ast.Constant, ast.Raise, ast.BinOp, ast.Div, ast.Mult,
                   ast.Add, ast.Sub, ast.Assign, ast.List, ast.For, ast.Attribute,
                   ast.ListComp, ast.comprehension, ast.Compare, ast.Eq, ast.USub)
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                return {'passed': None, 'note': 'Code outside restricted harness; needs review'}
            if isinstance(node, ast.Attribute) and node.attr != 'append':
                raise ValueError('Unexpected attribute')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in ('sum', 'len', 'ValueError'):
                raise ValueError('Unexpected function call')
        namespace = {'__builtins__': {'sum': sum, 'len': len, 'ValueError': ValueError}}
        exec(compile(tree, '<probe>', 'exec'), namespace)
        if name == 'python':
            fun = namespace['media']
            checks = [fun([1, 2, 3]) == 2, fun([-3, 1]) == -1, fun([2.5]) == 2.5]
            try:
                fun([])
                checks.append(False)
            except ValueError:
                checks.append(True)
        else:
            fun = namespace['quadrados']
            checks = [fun([1, 2, 3]) == [1, 4, 9], fun([]) == [], fun([-2, 0]) == [4, 0]]
        return {'passed': all(checks), 'checks': checks}
    if name == 'tool':
        obj = json.loads(answer)
        ok = obj.get('tool') == 'python_reviewer' and isinstance(obj.get('arguments'), dict) and all(k in obj['arguments'] for k in ('task', 'code')) and isinstance(obj.get('success_criteria'), list) and bool(obj['success_criteria'])
        return {'passed': ok, 'note': 'Schema and tool choice checked; task quality needs human review'}
    return {'passed': None, 'note': 'Qualitative review required'}

results = []
path = Path(os.environ.get('GPT_OSS_TEST_OUTPUT', str(Path(__file__).parent / 'auxiliary-tests.json')))
for name, prompt in CASES:
    started = time.perf_counter()
    item = {'test': name, 'prompt': prompt}
    try:
        body = json.dumps({'model': 'gpt-oss-20b', 'messages': [{'role': 'user', 'content': prompt}],
                           'max_tokens': 1536, 'temperature': 0, 'reasoning_effort': 'low', 'stream': False}).encode()
        headers = {'Content-Type': 'application/json', 'bypass-tunnel-reminder': 'true'}
        if os.environ.get('GPT_OSS_LOCAL_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['GPT_OSS_LOCAL_KEY']
        request = urllib.request.Request(URL, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
        item.update(model=data.get('model'), answer=data['choices'][0]['message'].get('content', ''),
                    finish_reason=data['choices'][0].get('finish_reason'), usage=data.get('usage'))
        try:
            item['verification'] = verify(name, item['answer'])
        except Exception as error:
            item['verification'] = {'passed': False, 'error': str(error)}
    except Exception as error:
        item['error'] = str(error)
    item['seconds'] = round(time.perf_counter() - started, 2)
    results.append(item)
    path.write_text(json.dumps({'endpoint': URL, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(item, ensure_ascii=True), flush=True)
    if 'error' in item:
        break
