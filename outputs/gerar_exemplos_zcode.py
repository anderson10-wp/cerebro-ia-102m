"""Gera exemplos programaticos (respostas calculadas por codigo) para a rodada local.
Ataca os erros observados: resposta unica com EOS, aritmetica, extracao de contexto,
JSON booleano, posicao de palavra, len() do Python. Nenhuma chamada de API.
Pula qualquer pergunta que ja exista no treino v5, nos professores ou nos conjuntos reservados.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rng = random.Random(20261007)

# --- conjuntos existentes: nada pode colidir ---
held = set()
for split in ['validation', 'audit']:
    for line in (ROOT / 'teacher-v3' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
        held.add(json.loads(line)['prompt'].strip().casefold())
seen = {k: '' for k in held}  # prompts reservados (validacao/auditoria) tambem sao proibidos no treino
for line in (ROOT / 'teacher-v5' / 'train.jsonl').read_text(encoding='utf-8-sig').splitlines():
    row = json.loads(line)
    seen[row['prompt'].strip().casefold()] = row['response']
for provider in ['groq', 'openrouter', 'gemini']:
    for batch in (ROOT / f'{provider}-teacher').glob('batch-*'):
        f = batch / 'train.jsonl'
        if not f.exists():
            continue
        for line in f.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            seen[row['prompt'].strip().casefold()] = row['response']

rows = []
used = set()

def add(category, question, response, scorer='exact'):
    prompt = f'Pergunta: {question}\nResposta:'
    key = prompt.strip().casefold()
    if key in used or key in seen:
        return False
    if scorer == 'json':
        json.loads(response)  # tem de ser JSON valido
    used.add(key)
    rows.append({'id': f'pgm-{len(rows):03d}', 'category': category, 'prompt': prompt,
                 'response': response, 'scorer': scorer, 'split': 'train',
                 'provenance': 'programmatic-local (zcode)', 'checked': True,
                 'validator': 'resposta calculada por codigo; exata ou JSON tipado'})
    return True

def pick(seq):
    return rng.choice(seq)

# --- 1. resposta unica de uma palavra (treino de EOS/parada) ---
palavras = ['azul', 'verde', 'amarelo', 'vermelho', 'roxo', 'preto', 'branco', 'cinza', 'rosa', 'marrom',
            'gato', 'cao', 'cavalo', 'tigre', 'leao', 'urso', 'coelho', 'pato', 'peixe', 'vaca',
            'maca', 'banana', 'uva', 'pera', 'manga', 'melancia', 'morango', 'limao', 'kiwi', 'figo',
            'copo', 'prato', 'faca', 'colher', 'mesa', 'cadeira', 'porta', 'janela', 'livro', 'caneta',
            'chave', 'caixa', 'bolsa', 'sapato', 'camisa', 'chapu', 'anel', 'relogio']
modelos_copia = ['Sua saída deve conter só: {w}',
                 'Responda apenas com a palavra {w}.',
                 'Copie exatamente uma vez, sem comentários: {w}',
                 'Escreva somente: {w}',
                 'A resposta deve ser apenas: {w}']
alvo = 80
tentativas = 0
while sum(1 for r in rows if r['category'] == 'instruction') < alvo and tentativas < 4000:
    tentativas += 1
    w = pick(palavras)
    add('instruction', pick(modelos_copia).format(w=w), w)

# --- 2. aritmetica ---
def aritmetica(alvo, gerador, frase, simbolo):
    feitos = set()
    tentativas = 0
    while sum(1 for r in rows if r['category'] == 'arithmetic') < alvo and tentativas < 6000:
        tentativas += 1
        a, b = gerador()
        if (a, b) in feitos:
            continue
        feitos.add((a, b))
        if not add('arithmetic', pick(frase).format(a=a, b=b, s=simbolo), str(a + b if simbolo == '+' else a * b if simbolo == '*' else a - b)):
            feitos.discard((a, b))

frases_soma = ['Escreva apenas o resultado de {a} + {b}.',
               'Quanto é {a} + {b}? Responda somente o número.',
               'Sem explicação, quanto vale {a} + {b}?',
               'Dê somente o valor de {a} + {b}.']
aritmetica(60, lambda: (rng.randint(11, 89), rng.randint(12, 99)), frases_soma, '+')
aritmetica(75, lambda: (rng.randint(3, 9), rng.randint(10, 49)), frases_soma, '+')
aritmetica(100, lambda: (rng.randint(2, 9), rng.randint(2, 9)),
           ['Quanto é {a} * {b}? Só o número.', 'Sem explicação, quanto vale {a} * {b}?',
            'Escreva apenas o resultado de {a} * {b}.'], '*')
aritmetica(125, lambda: (rng.randint(20, 99), rng.randint(5, 19)),
           ['Escreva apenas o resultado de {a} - {b}.', 'Quanto é {a} - {b}? Responda somente o número.'], '-')

# --- 3. extracao de contexto: bilhete ---
nomes = ['Rui', 'Ana', 'Léo', 'Bia', 'Gil', 'Eva', 'Ivo', 'Cris', 'Duda', 'Zeca', 'Nena', 'Toni']
objetos = [('chave', 'a'), ('celular', 'o'), ('caderno', 'o'), ('guarda-chuva', 'o'), ('óculos', 'os'),
           ('carteira', 'a'), ('livro', 'o'), ('cartão', 'o'), ('mochila', 'a'), ('boné', 'o')]
lugares = ['na gaveta', 'na mesa', 'na mochila', 'no bolso', 'na caixa', 'na prateleira',
          'no carro', 'na bolsa', 'na cozinha', 'na estante', 'na mesa de canto', 'na gaveta de baixo']
feitos = set()
tentativas = 0
while sum(1 for r in rows if r['category'] == 'context') < 60 and tentativas < 4000:
    tentativas += 1
    nome, (obj, art), lugar = pick(nomes), pick(objetos), pick(lugares)
    if (nome, obj, lugar) in feitos:
        continue
    feitos.add((nome, obj, lugar))
    if not add('context', f'Bilhete: {nome} deixou {art} {obj} {lugar}. Onde está {art} {obj}? Responda apenas o local.', lugar):
        feitos.discard((nome, obj, lugar))

# --- 4. ficha: dado presente vs ausente ---
campos = ['telefone', 'endereço', 'e-mail', 'cidade', 'documento']
tentativas = 0
while sum(1 for r in rows if r['category'] == 'context_missing') < 20 and tentativas < 2000:
    tentativas += 1
    n = pick(nomes)
    add('context_missing', f'Ficha: nome {n}; {pick(campos)} não registrado. Qual é o dado? Use "não informado" quando faltar.', 'não informado')
feitos = set()
tentativas = 0
while sum(1 for r in rows if r['category'] == 'context_value') < 20 and tentativas < 2000:
    tentativas += 1
    n = pick(nomes)
    idade = rng.randint(18, 80)
    if (n, idade) in feitos:
        continue
    if add('context_value', f'Ficha: nome {n}; idade {idade}. Qual é a idade? Responda apenas o número.', str(idade)):
        feitos.add((n, idade))

# --- 5. JSON booleano ---
chaves = ['ativo', 'ligado', 'pronto', 'ocupado', 'visivel', 'aberto', 'fechado', 'cheio', 'vazio', 'novo',
          'quente', 'frio', 'limpo', 'sujo', 'claro', 'escuro', 'forte', 'fraco', 'alto', 'baixo',
          'rapido', 'lento', 'caro', 'barato', 'facil', 'dificil', 'livre', 'usado', 'seco', 'molhado']
modelos_bool = ['Devolva somente JSON com a chave {k} e o booleano {b}.',
                'Use JSON para representar uma única chave {k} com valor booleano {b}. Nenhum outro campo.',
                'JSON com uma única chave {k}, valor booleano {b}, nada mais.']
feitos = set()
tentativas = 0
while sum(1 for r in rows if r['category'] == 'json') < 60 and tentativas < 4000:
    tentativas += 1
    k, b = pick(chaves), rng.random() < .5
    if (k, b) in feitos:
        continue
    rotulo = 'verdadeiro' if b else 'falso'
    if add('json', pick(modelos_bool).format(k=k, b=rotulo), json.dumps({k: b}, ensure_ascii=False, separators=(',', ':')), scorer='json'):
        feitos.add((k, b))

# --- 6. JSON de texto ---
feitos = set()
tentativas = 0
while sum(1 for r in rows if r['category'] == 'json_text') < 20 and tentativas < 2000:
    tentativas += 1
    k = pick(['nome', 'cor', 'cidade', 'animal', 'fruta', 'objeto', 'ferramenta', 'cliente'])
    v = pick(palavras)
    if (k, v) in feitos:
        continue
    if add('json_text', f'Devolva somente JSON com a chave {k} e o texto "{v}".',
           json.dumps({k: v}, ensure_ascii=False, separators=(',', ':')), scorer='json'):
        feitos.add((k, v))

# --- 7. posicao de palavra ---
def frase_palavras():
    return ' '.join(rng.sample(palavras, rng.choice([3, 4, 5])))
tentativas = 0
while sum(1 for r in rows if r['category'] == 'word_selection') < 100 and tentativas < 6000:
    tentativas += 1
    ws = frase_palavras().split()
    estilo = rng.random()
    if estilo < .4:
        pos = rng.randint(1, len(ws))
        q = f'Texto: {" ".join(ws)}. Escreva só a palavra número {pos}.'
    elif estilo < .65:
        q = f'Qual é a primeira palavra de "{" ".join(ws)}"? Escreva apenas essa palavra.'
        pos = 1
    elif estilo < .8:
        q = f'Escreva apenas a segunda palavra: {ws[0]} {ws[1]}.'
        ws = [ws[0], ws[1]]
        pos = 2
    else:
        q = f'Qual é a última palavra de "{" ".join(ws)}"? Escreva apenas ela.'
        pos = len(ws)
    add('word_selection', q, ws[pos - 1])

# --- 8. len() do Python ---
tentativas = 0
while sum(1 for r in rows if r['category'] == 'code') < 50 and tentativas < 2000:
    tentativas += 1
    tipo = rng.random()
    if tipo < .3:
        q, r_ = pick(['Qual é o resultado de len([]) em Python? Retorne somente o número.',
                      'Em Python, quanto vale len([])? Só o número.']), '0'
    elif tipo < .35:
        q, r_ = pick(['Qual é o resultado de len("") em Python? Retorne somente o número.',
                      'Em Python, quanto vale len("")? Só o número.']), '0'
    elif tipo < .65:
        itens = rng.randint(1, 9)
        lista = '[' + ', '.join(str(rng.randint(1, 99)) for _ in range(itens)) + ']'
        q, r_ = f'Em Python, qual o valor de len({lista})? Retorne somente o número.', str(itens)
    else:
        palavra = ''.join(rng.sample('abcdefghijlmnopqrstuvxz', rng.randint(2, 8)))
        q, r_ = f'Em Python, qual o valor de len("{palavra}")? Retorne somente o número.', str(len(palavra))
    add('code', q, r_)

# --- 9. simulacao de ferramenta ---
ferramentas = [('listar_arquivos', 'listar os arquivos'), ('obter_hora', 'saber a hora atual'),
               ('buscar_preco', 'consultar o preço'), ('contar_linhas', 'contar as linhas'),
               ('consultar_estoque', 'consultar o estoque'), ('agendar_reuniao', 'marcar a reunião'),
               ('verificar_saldo', 'verificar o saldo'), ('enviar_mensagem', 'enviar a mensagem')]
tentativas = 0
while sum(1 for r in rows if r['category'] == 'tool_simulation') < 25 and tentativas < 2000:
    tentativas += 1
    t, meta = pick(ferramentas)
    if rng.random() < .5:
        resposta = json.dumps({'ferramenta': t, 'argumentos': {}}, ensure_ascii=False, separators=(',', ':'))
        q = f'Exercício simulado: só existe {t}. Gere o objeto JSON {resposta} para {meta}. Não execute a consulta.'
    else:
        n = rng.randint(1, 99)
        resposta = json.dumps({'ferramenta': t, 'argumentos': {'id': n}}, ensure_ascii=False, separators=(',', ':'))
        q = f'Exercício simulado: só existe {t}. Gere o objeto JSON {resposta} para {meta} do item {n}. Não execute.'
    add('tool_simulation', q, resposta, scorer='json')
tentativas = 0
while sum(1 for r in rows if r['category'] == 'tool_result') < 25 and tentativas < 2000:
    tentativas += 1
    n = rng.randint(2, 999)
    if rng.random() < .5:
        q = f'Retorno simulado: {json.dumps({"resultado": n})}. Diga apenas o resultado numérico.'
    else:
        q = f'A ferramenta devolveu {json.dumps({"resultado": n})}. Qual é o número? Só o número.'
    add('tool_result', q, str(n))

# --- gravar ---
pasta = ROOT / 'programmatic-teacher' / 'batch-001'
pasta.mkdir(parents=True, exist_ok=True)
(pasta / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
(pasta / 'report.json').write_text(json.dumps({
    'status': 'completed', 'provider': 'programmatic-local',
    'note': 'Respostas calculadas por codigo (referencia deterministica). Sem chamadas de API.',
    'gerado_por': 'zcode', 'seed': 20261007, 'examples': len(rows)}, ensure_ascii=False, indent=2), encoding='utf-8')
por_categoria = {}
for r in rows:
    por_categoria[r['category']] = por_categoria.get(r['category'], 0) + 1
print(json.dumps({'gerados': len(rows), 'categorias': por_categoria}, ensure_ascii=False, indent=2))
