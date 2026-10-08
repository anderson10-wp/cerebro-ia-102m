"""Deterministic augmentation of reviewed teacher seeds, not fresh teacher output."""
import hashlib
import json
import random
from pathlib import Path
from tokenizers import Tokenizer
from routing import ROOT, BRAIN, TOOLS, SYSTEM, score, load_json, write_json

def prompt(case):
    return '<|system|>'+SYSTEM+'\nDisponíveis: '+', '.join(case['available'])+'<|user|>'+case['request']+'<|assistant|>'

def main():
    random.seed(20261004)
    tok = Tokenizer.from_file(str(BRAIN/'trained/brain/tokenizer.json'))
    cases = []
    def add(family, text, tool, args, available=None):
        row = {'id': 'expanded-'+str(len(cases)), 'family': family, 'split': 'train',
               'request': text, 'available': TOOLS if available is None else available,
               'expected': {'tool': tool, 'arguments': args}}
        assert score(row['expected'], row)['correct']
        cases.append(row)
    math_templates = ['Calcule {}.', 'Quanto é {}?', 'Resolva {}.', 'Preciso do resultado de {}.']
    for op in ['+', '-']:
        for template in math_templates:
            for _ in range(16):
                a, b = random.sample(range(12, 980), 2)
                expression = f'{a} {op} {b}'
                add('arithmetic_add_sub', template.format(expression), 'calculator', {'expression': expression})
    subjects = ['uma variável em Python', 'um dicionário em Python', 'uma lista em Python', 'um laço for',
                'uma função Python', 'uma string', 'um arquivo CSV', 'um arquivo JSON',
                'uma API HTTP', 'um servidor web', 'memória RAM', 'armazenamento SSD',
                'um sistema operacional', 'uma pasta de arquivos', 'um endereço IP', 'um navegador web']
    explanation_templates = ['Explique o que é {}.', 'Descreva {} em português.', 'Dê uma explicação simples sobre {}.',
         'Me ensine o básico sobre {}.', 'Escreva uma introdução a {}.', 'O que significa {}?',
         'Quero entender melhor {}.', 'Dê um exemplo de uso de {}.']
    for topic in subjects:
        for template in explanation_templates:
            request = template.format(topic)
            add('basic_explanation', request, 'teacher', {'request': request})
    greetings = ['', 'Por favor, ', 'Oi, ', 'Olá, ', 'Bom dia, ', 'Boa tarde, ', 'Boa noite, ',
                 'Quando puder, ', 'Agora, ', 'Neste momento, ', 'Para eu acompanhar, ',
                 'Me ajude: ', 'Estou acompanhando o projeto: ', 'Sobre nosso modelo: ',
                 'Pode conferir: ', 'Gostaria de saber: ']
    status_requests = ['qual o progresso real do nosso treinamento?', 'nosso treino está em execução?',
        'quantos tokens nosso modelo já processou no treino?', 'o treino do nosso modelo terminou?',
        'a execução atual de treinamento concluiu?', 'o treinamento ainda está rodando?',
        'o objetivo de tokens do nosso treino foi alcançado?', 'o treino do 102M está ativo agora?']
    for greeting in greetings:
        for request in status_requests:
            add('progress_completion', greeting+request, 'training_status', {})
    missing = [('analise o arquivo que ainda não enviei.', 'Pode enviar o arquivo para análise?'),
               ('corrija o código que não anexei.', 'Pode enviar o código e descrever o problema?'),
               ('resuma o texto que ainda vou mandar.', 'Pode enviar o texto para eu resumir?'),
               ('compare os documentos que não anexei.', 'Pode enviar os documentos que deseja comparar?')]
    for greeting in greetings:
        for request, question in missing:
            add('missing_artifact', greeting+request, 'clarify', {'question': question})
    for index, topic in enumerate(subjects):
        for template in explanation_templates[:2]:
            request = template.format(topic)
            add('teacher_unavailable', request, 'clarify',
                {'question': 'O professor está indisponível. Você quer aguardar para continuar?'},
                ['calculator', 'training_status', 'clarify'])
    for greeting in greetings[:4]:
        for request in status_requests:
            add('status_unavailable', greeting+request, 'clarify',
                {'question': 'A consulta ao treino está indisponível. Você pode fornecer o relatório atual?'},
                ['calculator', 'teacher', 'clarify'])
    seed = [json.loads(line) for line in (ROOT/'sft-train.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = list(seed)
    seen = {r['prompt'] for r in rows}
    for case in cases:
        p = prompt(case)
        if p in seen:
            continue
        seen.add(p)
        rows.append({'id': case['id'], 'family': case['family'], 'prompt': p,
            'response': json.dumps(case['expected'], ensure_ascii=False, separators=(',', ':')),
            'label_source': 'codex_deterministic_augmentation_of_reviewed_protocol'})
    heldout = []
    for split in ('validation', 'audit'):
        heldout.extend(json.loads(line) for line in (ROOT/f'sft-{split}.jsonl').read_text(encoding='utf-8').splitlines())
    assert not seen & {r['prompt'] for r in heldout}
    for row in rows:
        p = tok.encode(row['prompt'], add_special_tokens=False).ids
        r = tok.encode(row['response']+'<|eos|>', add_special_tokens=False).ids
        assert len(p)+len(r) <= 1024
        row.update({'prompt_tokens': len(p), 'response_tokens': len(r)})
    random.shuffle(rows)
    path = ROOT/'pilot-train.jsonl'
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')
    manifest = {'training_examples': len(rows), 'reviewed_teacher_seeds': len(seed),
        'deterministic_augmented_examples': len(rows)-len(seed),
        'validation_examples': 16, 'audit_examples': 16,
        'total_tokens': sum(r['prompt_tokens']+r['response_tokens'] for r in rows),
        'supervised_response_tokens': sum(r['response_tokens'] for r in rows),
        'train_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'max_sequence_tokens': max(r['prompt_tokens']+r['response_tokens'] for r in rows),
        'scope': 'A narrow tool-routing pilot. Most added examples are programmatic variations, not independent natural conversations.',
        'selection_rule': 'Select checkpoint by validation loss only; evaluate frozen audit once after selection.',
        'continuation_gate': {'minimum_audit_correct_of_16': 12, 'minimum_validation_correct_of_16': 12,
                              'minimum_audit_schema_valid_of_16': 14, 'maximum_regression_nll_ratio': 1.10},
        'gate_note': 'Engineering thresholds chosen before tuning, not statistically proven guarantees. Passing allows considering a broader test, not autonomous execution.'}
    write_json(ROOT/'pilot-data-manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=True), flush=True)

if __name__ == '__main__':
    main()
