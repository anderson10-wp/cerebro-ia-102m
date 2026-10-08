"""Small teacher-data and CPU-baseline pilot; never executes a proposed tool."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
BRAIN = ROOT.parent / 'cerebro-v1'
TOOLS = ['calculator', 'teacher', 'training_status', 'clarify']
SYSTEM = ('Escolha UMA ferramenta. Responda somente JSON: '
          '{"tool":"nome","arguments":{...}}. '
          'calculator: expression copiada do pedido; teacher: request igual ao pedido; '
          'training_status: argumentos vazios; clarify: question pedindo informação que falta. '
          'Use calculator para aritmética, teacher para explicações/código, training_status '
          'para estado real do nosso treino. Se falta informação ou a ferramenta necessária '
          'está indisponível, use clarify. Não execute ferramentas nem invente resultados.')

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def seeds():
    rows = []
    def add(split, family, tool, prompts, expressions=None, available=None):
        for index, prompt in enumerate(prompts):
            args = {'expression': expressions[index]} if tool == 'calculator' else (
                {'request': prompt} if tool == 'teacher' else (
                    {} if tool == 'training_status' else {'question': 'Pode fornecer a informação que falta ou escolher uma ferramenta disponível?'}))
            rows.append({'id': f'{split}-{family}-{index}', 'split': split, 'family': family,
                         'request': prompt, 'available': available if available is not None else TOOLS,
                         'expected': {'tool': tool, 'arguments': args}})
    add('train', 'addition', 'calculator',
        ['Calcule 7 + 8.', 'Quanto é 19 + 26?', 'Resolva 105 + 29.', 'Calcule 34 + 68.'],
        ['7 + 8', '19 + 26', '105 + 29', '34 + 68'])
    add('train', 'subtraction', 'calculator',
        ['Calcule 23 - 9.', 'Quanto é 51 - 18?', 'Resolva 97 - 32.', 'Calcule 200 - 74.'],
        ['23 - 9', '51 - 18', '97 - 32', '200 - 74'])
    add('train', 'python_creation', 'teacher',
        ['Escreva uma função Python que some dois números.', 'Crie uma função Python que calcule a média de uma lista.',
         'Escreva código Python para contar palavras.', 'Crie uma função Python que encontre o maior número.'])
    add('train', 'text_explanation', 'teacher',
        ['Explique a diferença entre memória RAM e armazenamento.', 'Explique o que é uma variável em Python.',
         'Explique o que significa uma API.', 'Explique como funciona uma lista em Python.'])
    add('train', 'progress', 'training_status',
        ['Qual é o progresso real do nosso treinamento?', 'O treino do nosso modelo está rodando?',
         'Quantos tokens nosso modelo já treinou?', 'Mostre o estado atual do nosso treino.'])
    add('train', 'completion', 'training_status',
        ['O treinamento do nosso 102M terminou?', 'A execução de treino chegou ao objetivo?',
         'Nosso treino finalizou sem erro?', 'A terceira sessão de treino já concluiu?'])
    add('train', 'missing_artifact', 'clarify',
        ['Corrija o código que esqueci de enviar.', 'Analise o arquivo; ainda não o anexei.',
         'Resuma o texto que ainda vou mandar.', 'Compare os dois documentos que não enviei.'])
    add('train', 'teacher_unavailable', 'clarify',
        ['Explique o que é uma GPU.', 'Crie uma função Python de ordenação.',
         'Explique redes neurais.', 'Escreva um programa para ler CSV.'],
        available=['calculator', 'training_status', 'clarify'])
    add('validation', 'multiplication', 'calculator',
        ['Calcule 8 * 7.', 'Quanto é 13 * 4?', 'Resolva 17 * 6.', 'Calcule 21 * 9.'],
        ['8 * 7', '13 * 4', '17 * 6', '21 * 9'])
    add('validation', 'model_architecture', 'teacher',
        ['Explique atenção em um transformer.', 'Explique o que são parâmetros de um modelo.',
         'Explique a função do tokenizador.', 'Explique para que serve uma camada neural.'])
    add('validation', 'training_metrics', 'training_status',
        ['Qual foi a loss final do nosso treinamento?', 'Qual foi o resultado de validação do nosso treino?',
         'Qual o número de passos registrado no treino?', 'Qual foi a duração acumulada do treino?'])
    add('validation', 'calculator_unavailable', 'clarify',
        ['Calcule 9 + 2.', 'Quanto é 14 * 3?', 'Resolva 48 - 7.', 'Calcule 81 / 9.'],
        available=['teacher', 'training_status', 'clarify'])
    add('audit', 'parenthesized_arithmetic', 'calculator',
        ['Calcule (12 + 6) / 3.', 'Quanto é (8 + 2) * 5?', 'Resolva (30 - 12) / 6.', 'Calcule (7 * 4) - 9.'],
        ['(12 + 6) / 3', '(8 + 2) * 5', '(30 - 12) / 6', '(7 * 4) - 9'])
    add('audit', 'experiment_design', 'teacher',
        ['Como separar exemplos de treinamento e teste?', 'Como verificar se uma nova versão realmente melhorou?',
         'Como evitar que um modelo memorize a avaliação?', 'Como construir testes para chamadas de ferramentas?'])
    add('audit', 'checkpoint_and_remaining_time', 'training_status',
        ['Qual checkpoint foi selecionado no nosso treino?', 'Quanto tempo falta para concluir nosso treino?',
         'Em qual sessão nosso treino parou?', 'O arquivo final de pesos do nosso treino foi salvo?'])
    add('audit', 'unspecified_reference', 'clarify',
        ['Melhore isso.', 'Resolva o problema.', 'Faça aquela alteração.', 'Compare as opções.'])
    return rows

def prepare():
    rows = seeds()
    seen = set()
    families = {}
    for row in rows:
        key = ' '.join(row['request'].lower().split())
        if key in seen:
            raise ValueError('Duplicate request')
        seen.add(key)
        families.setdefault(row['split'], set()).add(row['family'])
    splits = list(families)
    for i, split in enumerate(splits):
        for other in splits[i+1:]:
            if families[split] & families[other]:
                raise ValueError('Task family crossed splits')
    write_json(ROOT / 'routing-cases.json', rows)
    write_json(ROOT / 'routing-manifest.json', {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'counts': {split: sum(r['split'] == split for r in rows) for split in splits},
        'family_policy': 'Manually separated task categories; normalized exact requests are distinct. This is a small probe, not a broad benchmark.',
        'teacher_policy': 'Teacher sees training requests only; validation/audit labels are independently specified.',
        'weights_policy': 'Use the existing scratch 102M weights and tokenizer; no pretrained principal weights.',
        'tool_policy': 'Outputs are requests only. No tool is executed by this pilot.'})
    print(json.dumps({'status': 'seed_ready', 'cases': len(rows)}), flush=True)

def validate_route(value, case):
    if not isinstance(value, dict) or set(value) != {'tool', 'arguments'}:
        raise ValueError('Expected only tool and arguments')
    tool, args = value['tool'], value['arguments']
    if tool not in TOOLS or tool not in case['available']:
        raise ValueError('Unknown or unavailable tool')
    if not isinstance(args, dict):
        raise ValueError('Arguments must be an object')
    fields = {'calculator': {'expression'}, 'teacher': {'request'}, 'training_status': set(), 'clarify': {'question'}}
    if set(args) != fields[tool]:
        raise ValueError('Invalid arguments for tool')
    for text in args.values():
        if not isinstance(text, str) or not text.strip() or len(text) > 300:
            raise ValueError('Invalid argument text')
    if tool == 'calculator' and ''.join(args['expression'].split()) not in ''.join(case['request'].split()):
        raise ValueError('Expression was not copied from request')
    if tool == 'teacher' and args['request'] != case['request']:
        raise ValueError('Teacher request must preserve the original request')
    return value

def score(value, case):
    try:
        validate_route(value, case)
    except (ValueError, TypeError) as error:
        return {'schema_valid': False, 'correct': False, 'reason': str(error)}
    expected = case['expected']
    correct = value['tool'] == expected['tool']
    if correct and value['tool'] == 'calculator':
        correct = ''.join(value['arguments']['expression'].split()) == ''.join(expected['arguments']['expression'].split())
    if correct and value['tool'] == 'teacher':
        correct = value['arguments'] == expected['arguments']
    return {'schema_valid': True, 'correct': correct,
            'reason': None if correct else 'Wrong tool or arguments',
            'semantic_question_review_needed': value['tool'] == 'clarify'}

def teacher():
    rows = [r for r in load_json(ROOT / 'routing-cases.json') if r['split'] == 'train']
    endpoint = load_json(ROOT.parent / 'gpt-oss-kaggle/endpoint.json')['base_url']
    submitted = {'type': 'function', 'function': {'name': 'submit_routes',
        'description': 'Submit one routing decision for each provided request, without executing anything.',
        'parameters': {'type': 'object', 'properties': {'routes': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'id': {'type': 'string'},
                'tool': {'type': 'string', 'enum': TOOLS},
                'arguments': {'type': 'object', 'properties': {
                    'expression': {'type': 'string'}, 'request': {'type': 'string'}, 'question': {'type': 'string'}},
                    'additionalProperties': False}},
            'required': ['id', 'tool', 'arguments'], 'additionalProperties': False}}},
            'required': ['routes'], 'additionalProperties': False}}}
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'model': 'gpt-oss-20b',
              'split_seen_by_teacher': 'train', 'batches': [], 'results': [], 'paid_api_calls': 0}
    for offset in range(0, len(rows), 8):
        cases = rows[offset:offset+8]
        body = {'model': 'gpt-oss-20b', 'temperature': 0, 'max_tokens': 2048, 'reasoning_effort': 'low',
                'messages': [{'role': 'system', 'content': SYSTEM + ' Envie as decisões usando submit_routes.'},
                             {'role': 'user', 'content': json.dumps([
                                 {k: r[k] for k in ('id', 'request', 'available')} for r in cases], ensure_ascii=False)}],
                'tools': [submitted], 'tool_choice': {'type': 'function', 'function': {'name': 'submit_routes'}}}
        req = urllib.request.Request(endpoint + '/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['GPT_OSS_LOCAL_KEY']})
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=150) as response:
            data = json.load(response)
        choice = data['choices'][0]
        calls = choice['message'].get('tool_calls') or []
        if len(calls) != 1 or calls[0]['function']['name'] != 'submit_routes':
            raise ValueError('Teacher did not use the expected tool')
        arguments = json.loads(calls[0]['function']['arguments'])
        if set(arguments) != {'routes'} or not isinstance(arguments['routes'], list):
            raise ValueError('Invalid batch envelope')
        decisions = arguments['routes']
        by_id = {r['id']: r for r in cases}
        if len(decisions) != len(cases) or {r['id'] for r in decisions} != set(by_id):
            raise ValueError('Missing, repeated, or unexpected case IDs')
        for decision in decisions:
            if set(decision) != {'id', 'tool', 'arguments'}:
                raise ValueError('Unexpected route fields')
            route = {k: decision[k] for k in ('tool', 'arguments')}
            result = score(route, by_id[decision['id']])
            report['results'].append({'id': decision['id'], 'route': route, 'verification': result})
        report['batches'].append({'count': len(cases), 'seconds': round(time.perf_counter()-started, 2),
                                 'finish_reason': choice.get('finish_reason'), 'usage': data.get('usage')})
        write_json(ROOT / 'teacher-routing-results.json', report)
        print(json.dumps({'batch': offset//8+1, 'done': len(report['results']),
                          'correct': sum(r['verification']['correct'] for r in report['results'])}), flush=True)
    report['status'] = 'awaiting_question_review'
    write_json(ROOT / 'teacher-routing-results.json', report)

def baseline():
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(BRAIN))
    from model import Brain, Config
    torch.set_num_threads(4)
    torch.manual_seed(42)
    folder = BRAIN / 'trained/brain'
    checkpoint = folder / 'best-model.pt'
    tok = Tokenizer.from_file(str(folder / 'tokenizer.json'))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = Brain(Config(**saved['config']))
    model.load_state_dict(saved['model'])
    model.eval()
    rows = [r for r in load_json(ROOT / 'routing-cases.json') if r['split'] == 'audit']
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'checkpoint': str(checkpoint), 'checkpoint_tokens': saved.get('tokens'),
              'device': 'cpu', 'decode': 'greedy, no grammar constraints', 'max_new_tokens': 96,
              'split': 'audit', 'results': [], 'tool_execution_count': 0}
    eos = tok.token_to_id('<|eos|>')
    for case in rows:
        prompt = '<|system|>' + SYSTEM + '\nDisponíveis: ' + ', '.join(case['available']) + '<|user|>' + case['request'] + '<|assistant|>'
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids])
        if ids.shape[1] + report['max_new_tokens'] > model.cfg.context:
            raise ValueError('Prompt would exceed context')
        outputs = []
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(report['max_new_tokens']):
                next_id = int(model(ids)[:, -1].argmax(-1).item())
                if next_id == eos:
                    break
                outputs.append(next_id)
                ids = torch.cat((ids, torch.tensor([[next_id]])), dim=1)
                text = tok.decode(outputs)
                try:
                    route = json.loads(text)
                    if isinstance(route, dict):
                        break
                except ValueError:
                    pass
        output = tok.decode(outputs)
        try:
            route = json.loads(output)
            verification = score(route, case)
        except ValueError:
            verification = {'schema_valid': False, 'correct': False, 'reason': 'Not a complete JSON response'}
        report['results'].append({'id': case['id'], 'request': case['request'], 'output': output,
                                  'generated_tokens': len(outputs), 'verification': verification,
                                  'seconds': round(time.perf_counter()-started, 2)})
        write_json(ROOT / 'routing-baseline.json', report)
        print(json.dumps({'done': len(report['results']), 'total': len(rows),
                          'correct': sum(r['verification']['correct'] for r in report['results']),
                          'seconds': report['results'][-1]['seconds']}), flush=True)
    report['status'] = 'complete'
    report['summary'] = {'cases': len(rows),
        'schema_valid': sum(r['verification']['schema_valid'] for r in report['results']),
        'correct': sum(r['verification']['correct'] for r in report['results']),
        'mean_seconds': round(sum(r['seconds'] for r in report['results'])/len(rows), 2)}
    write_json(ROOT / 'routing-baseline.json', report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'teacher', 'baseline'])
    args = parser.parse_args()
    {'prepare': prepare, 'teacher': teacher, 'baseline': baseline}[args.mode]()
