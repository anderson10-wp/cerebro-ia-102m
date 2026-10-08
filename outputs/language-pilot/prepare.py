"""Small curated Portuguese SFT and frozen diagnostic; no teacher API calls."""
import hashlib,json,random,re
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent; OUT=ROOT.parent

FACTS=[
('Brasil','Brasília','Brasília é a capital do Brasil.'),
('Portugal','Lisboa','Lisboa é a capital de Portugal.'),
('França','Paris','Paris é a capital da França.'),
('Argentina','Buenos Aires','Buenos Aires é a capital da Argentina.'),
('Japão','Tóquio','Tóquio é a capital do Japão.'),
('Itália','Roma','Roma é a capital da Itália.'),
('Alemanha','Berlim','Berlim é a capital da Alemanha.'),
('Espanha','Madri','Madri é a capital da Espanha.')]
SEEDS=[
('Olá!','Olá! Como posso ajudar você hoje?'),
('O que você pode fazer?','Posso ajudar a interpretar textos, responder perguntas e organizar informações. Minhas respostas podem conter erros; informações importantes precisam ser verificadas.'),
('Não entendi sua resposta.','Vou tentar explicar de maneira mais simples. Qual parte ficou confusa?'),
('Você pode me ajudar a estudar?','Sim. Diga qual assunto você quer estudar e o que já conhece. Podemos começar com uma explicação curta e depois praticar com exemplos.'),
('Quero organizar minhas tarefas.','Escreva as tarefas que precisa fazer. Depois podemos separar as mais urgentes, estimar o tempo e escolher uma ordem para começar.'),
('Estou testando você.','Tudo bem. Envie uma pergunta ou uma instrução específica. Vou tentar responder ao que você pediu.'),
('Responda com uma saudação curta.','Olá! Como posso ajudar?'),
('Você tem certeza de tudo que diz?','Não. Posso cometer erros ou não ter informações suficientes. Quando houver dúvida, é melhor reconhecer a incerteza e verificar.'),
('Você está acessando a internet nesta conversa?','Não há uma ferramenta de internet conectada a esta conversa. Eu não posso afirmar que consultei uma página ou executei uma ação externa.'),
('Como você sabe se executou um comando?','É necessário receber o resultado de uma ferramenta que realmente executou o comando. Escrever um comando em uma resposta não significa que ele foi executado.'),
('Qual é a diferença entre escrever um comando e executá-lo?','Escrever um comando produz texto. Executá-lo exige uma ferramenta ou programa que realize a ação e devolva um resultado.'),
('A capital do Brasil é São Paulo?','Não. A capital do Brasil é Brasília. São Paulo é a capital do estado de São Paulo.'),
('As plantas não precisam de água?','As plantas precisam de água. A água participa de processos importantes para seu crescimento e funcionamento.'),
('O que é fotossíntese?','Na fotossíntese, plantas usam energia da luz para produzir açúcares a partir de água e dióxido de carbono. Esse processo também libera oxigênio.'),
('O que é um verbo?','Um verbo é uma palavra que pode indicar ação, estado ou fenômeno. Correr, estar e chover são exemplos.'),
('O que é um substantivo?','Um substantivo nomeia seres, objetos, lugares ou ideias. Casa, cidade e alegria são exemplos.'),
('O que é uma pergunta?','Uma pergunta solicita uma informação ou um esclarecimento. Em português escrito, muitas perguntas terminam com ponto de interrogação.'),
('O que é um polígono?','Um polígono é uma figura plana fechada formada por segmentos de reta. Triângulos e quadrados são exemplos.'),
('O que é um triângulo?','Um triângulo é um polígono que tem três lados.'),
('O que é um quadrado?','Um quadrado tem quatro lados de mesmo comprimento e quatro ângulos retos.'),
('O que são números pares?','Números pares são inteiros divisíveis por dois sem deixar resto. Dois, quatro e seis são exemplos.'),
('O que é um arquivo?','Um arquivo é uma unidade de dados armazenada em um dispositivo. Ele pode conter texto, imagens, código ou outros tipos de informação.'),
('O que é uma pasta no computador?','Uma pasta serve para organizar arquivos e outras pastas. Ela ajuda a encontrar e separar os conteúdos armazenados.'),
('O que é um programa?','Um programa é um conjunto de instruções que um computador pode executar para realizar tarefas.'),
('O que é uma variável em programação?','Uma variável associa um nome a um valor. O programa pode usar esse nome para acessar ou atualizar o valor.'),
('O que é um erro em um programa?','É um problema que pode impedir a execução ou produzir um resultado incorreto. A mensagem de erro e os dados de entrada ajudam a investigar a causa.'),
('O que é um conjunto de dados?','É uma coleção organizada de exemplos ou registros. Em aprendizado de máquina, ele pode ser usado para treinar ou avaliar um modelo.'),
('Treinar um modelo é a mesma coisa que testá-lo?','Não. O treinamento ajusta os parâmetros usando exemplos. O teste mede o comportamento em exemplos reservados que não devem orientar esses ajustes.'),
('Uma loss menor prova que a IA responde tudo corretamente?','Não. Uma loss menor indica melhora naquele objetivo e conjunto de dados. É necessário verificar também as respostas em testes separados.'),
('O que é um checkpoint?','É um arquivo que registra os pesos de um modelo em um ponto do treinamento. Ele permite guardar uma versão e, quando inclui o estado necessário, retomar o processo.'),
('O que é uma GPU?','Uma GPU é um processador que executa muitas operações em paralelo. Ela pode acelerar cálculos usados no treinamento e na execução de modelos.'),
('O que é um tokenizer?','Um tokenizer transforma texto em unidades chamadas tokens. O modelo usa os identificadores desses tokens para processar e gerar texto.'),
('Mais tokens vistos significam mais dados diferentes?','Não necessariamente. Um treinamento pode passar várias vezes pelos mesmos textos. A contagem de tokens processados inclui essas repetições.'),
('Uma resposta longa é sempre uma resposta melhor?','Não. A resposta precisa estar correta, seguir a pergunta e ter o tamanho adequado. Um texto longo pode repetir informações ou sair do assunto.'),
('Como começar a estudar um assunto novo?','Comece com os conceitos básicos e uma explicação simples. Faça pequenos exercícios e identifique o que ainda não entendeu antes de avançar.'),
('Como escrever uma mensagem clara?','Diga o objetivo da mensagem, apresente as informações necessárias e use frases diretas. Revise o texto para retirar ambiguidades e repetições.'),
('Por que precisamos de testes separados?','Se avaliarmos apenas os exemplos usados no treino, podemos confundir memorização com aprendizado. Testes separados ajudam a verificar o comportamento em situações novas.'),
('Se o texto não informa a resposta, o que você deve fazer?','Devo dizer que a informação não foi fornecida, em vez de inventar uma resposta.'),
('Como resumir um texto?','Identifique a ideia principal e as informações essenciais. Escreva uma versão mais curta, preservando o significado e sem acrescentar fatos.'),
('O que significa comparar dois resultados?','Significa observar as diferenças usando o mesmo critério. Para avaliar uma mudança, precisamos conhecer tanto o resultado anterior quanto o atual.')]

def prompt(q): return 'Pergunta: '+q+'\nResposta:'
def main():
    tok=Tokenizer.from_file(str(OUT/'h100-pretrain/result/tokenizer.json')); rng=random.Random(81802)
    names=['Ana','Bruno','Carla','Diego','Elisa','Fábio','Gabriela','Hugo','Iara','João','Lia','Marcos','Nina','Otávio','Paula','Rui','Sara','Tiago','Vera','Yuri']
    places=['na estante','na gaveta','sobre a mesa','dentro da mochila','no armário','perto da janela','na caixa','ao lado da porta']
    objects=['livro','caderno','mapa','relógio','lápis','cartão','bilhete','controle','fone','envelope']
    words=['janela','vento','folha','nuvem','ponte','campo','estrela','tinta','pedra','música','pássaro','água','árvore','papel','areia','chave','luz','praia','jardim','caneta','círculo','porta','tempo','palavra']
    train=[]; dev=[]; tests=[]; seen=set()
    def add(dst,family,q,a,scorer='exact',pair_id=None):
        p=prompt(q); key=p.casefold().strip()
        if key in seen: return
        seen.add(key)
        row={'id':f'{family}-{len(dst)}','category':family,'prompt':p,'response':a,'scorer':scorer,'quality':'original verified construction or authored review'}
        if pair_id: row['pair_id']=pair_id
        assert len(tok.encode(p).ids+tok.encode(a+'<|eos|>').ids)<=384
        dst.append(row)
    # Varied full sentences grounded entirely in fictional source text.
    for i in range(360):
        who=rng.choice(names); where=rng.choice(places); obj=rng.choice(objects); other=rng.choice([n for n in names if n!=who]); number=rng.randrange(2,40)
        contexts=[f'{who} guardou o {obj} {where}. {other} chegou depois.',f'O {obj} foi colocado {where} por {who}. {other} trouxe {number} folhas.',f'Ao voltar, {who} deixou o {obj} {where}. Mais tarde, {other} telefonou.']
        context=contexts[i%3]
        forms=[f'Leia: {context} Onde ficou o {obj}? Responda em uma frase.',f'Com base neste texto, informe onde está o {obj}: {context}',f'Texto: {context} Localize o {obj} e responda com uma frase curta.']
        add(train,'grounded_sentence',forms[i%3],f'O {obj} ficou {where}.')
        add(train,'grounded_who',f'Trecho: {context} Quem guardou o {obj}? Escreva apenas o nome.',who)
        add(train,'grounded_unknown',f'Trecho: {context} Em que ano {who} nasceu? Se o trecho não disser, escreva não informado.','não informado')
        add(train,'grounded_summary',f'Resuma o acontecimento em duas frases curtas, sem acrescentar fatos: {who} guardou o {obj} {where}. {other} chegou depois. Os dois conversaram.',f'{who} guardou o {obj} {where}. Depois, {other} chegou e os dois conversaram.')
        if i%2==0:
            add(train,'grounded_dialogue',f'Conversa: eu perguntei onde está o {obj}. Você perguntou onde eu o deixei. Eu respondi que o deixei {where}. Agora responda à minha pergunta inicial com uma frase.',f'Você deixou o {obj} {where}.')
    for i in range(160):
        who=rng.choice(names); other=rng.choice([n for n in names if n!=who]); obj=rng.choice(objects); where=rng.choice(places)
        add(train,'grounded_negation',f'{who} não guardou o {obj}. Quem guardou foi {other}, {where}. Quem guardou o {obj}? Escreva somente o nome.',other)
        add(train,'grounded_boolean',f'No texto, {who} não tem um {obj}, mas {other} tem. {who} tem um {obj}? Responda sim ou não.','não')
        add(train,'grounded_boolean',f'No texto, {who} não tem um {obj}, mas {other} tem. {other} tem um {obj}? Responda sim ou não.','sim')
    for i in range(240):
        word=rng.choice(words); obj=rng.choice(objects); who=rng.choice(names)
        forms=[f'Copie apenas a palavra {word}.',f'Escreva exatamente: {word}',f'Sua resposta deve ser somente {word}.',f'Repita uma vez, sem explicação: {word}']
        add(train,'copy',forms[i%4],word)
        text=f'{who} encontrou um {obj}.'
        add(train,'lowercase',f'Coloque em letras minúsculas, sem mudar as palavras: {text}',text.lower())
        add(train,'json',f'Gere somente um JSON com a chave palavra e o valor {word}.',json.dumps({'palavra':word},ensure_ascii=False,separators=(',',':')),'json')
    for i in range(100):
        a=rng.randrange(0,21); b=rng.randrange(0,10)
        add(train,'arithmetic',f'Some {a} e {b}. Responda só o número.',str(a+b))
    # A small authored language component, not blind dataset imports.
    for q,a in SEEDS:
        add(train,'reviewed_language',q,a)
        add(train,'reviewed_language',f'Explique de maneira simples: {q}',a)
    for country,capital,sentence in FACTS:
        for q in [f'Qual é a capital de {country}?',f'Diga qual cidade é a capital de {country}.',f'Explique em uma frase qual é a capital de {country}.']:
            add(train,'reviewed_fact',q,sentence)
    # Two inspected Hugging Face constraints, with corrected/reviewed responses.
    add(train,'hf_reviewed','Sua resposta deve conter menos de 50 palavras. Não use proibido, restrito, não permitido, vetado ou banido. Termine com a frase exata: Posso ajudar em mais alguma coisa?','Claro, diga qual é a sua dúvida. Posso ajudar em mais alguma coisa?')
    add(train,'hf_reviewed','Escreva exatamente cinco pontos usando marcadores *. Use somente letras minúsculas.','* primeiro ponto\n* segundo ponto\n* terceiro ponto\n* quarto ponto\n* quinto ponto')
    for r in train:
        if r['category']=='hf_reviewed': r.update(source='marlosb/smol-smoltalk-pt',revision='fe468af00de1f8d948a40399755e1faae0fa9cda',license='Apache-2.0 declared by publisher',modified=True)
    # Freeze 64 objective tests and 8 open prompts BEFORE training.
    for dst,split in [(dev,'dev'),(tests,'test')]:
        for family in ['copy','context','negation','missing','case','json','arithmetic','fact']:
            for pair in range(4):
                for v in [0,1]:
                    idx=pair*2+v; who=names[(idx+(10 if split=='test' else 2))%len(names)]; obj=objects[idx]; word=words[idx+(12 if split=='test' else 0)]
                    if family=='copy':
                        word=(word+' '+who.lower()) if split=='test' else word+' '+str(idx)
                        q=f'Devolva somente o texto entre colchetes: [{word}]'; answer=word
                    elif family=='context':
                        place=places[idx]; other=places[(idx+3)%len(places)]
                        q=f'Registro: o {obj} está {place}. A mochila está {other}. Qual é o local do {obj}? Responda apenas o local.'; answer=place
                    elif family=='negation':
                        owner=names[(idx+5)%len(names)]
                        q=f'{who} não levou o {obj}; {owner} levou. A pergunta é sobre {obj}: quem levou? Responda somente o nome.'; answer=owner
                    elif family=='missing':
                        date=str(1990+idx+(10 if split=='test' else 0))
                        context=f'{who} gosta de livros.'+(f' Nasceu em {date}.' if v else '')
                        q=f'Informações disponíveis: {context} Qual o ano de nascimento? Escreva só o ano, ou não informado se ele não aparece.'; answer=date if v else 'não informado'
                    elif family=='case':
                        text=f'{who} LEU o {obj}.'; q=f'Reescreva todo este texto com letras minúsculas: {text}'; answer=text.lower()
                    elif family=='json':
                        q=f'Devolva um objeto JSON cuja única chave seja item e cujo valor seja "{word}".'; answer=json.dumps({'item':word},ensure_ascii=False)
                    elif family=='arithmetic':
                        a=31+idx if split=='test' else 22+idx; b=2+v; q=f'Quanto resulta de {a} mais {b}? Retorne somente o resultado numérico.'; answer=str(a+b)
                    else:
                        country,capital,sentence=FACTS[idx]
                        q=f'Qual cidade é a capital de {country}? Escreva somente o nome da cidade.' if split=='test' else f'Complete com uma cidade: a capital de {country} é'; answer=capital
                    if split=='dev': q='Tarefa curta: '+q
                    add(dst,family,q,answer,'json' if family=='json' else 'exact',f'{split}-{family}-{pair}')
        assert len(dst)==64
    open_questions=[
      'Oi! Quero conversar com você. Como pode me ajudar?',
      'Você afirmou que São Paulo é a capital do Brasil. Essa informação está correta?',
      'Por que as plantas precisam de água? Explique de maneira simples.',
      'Estou estudando um assunto novo e não sei por onde começar. Sugira um primeiro passo.',
      'Explique a diferença entre treinar um modelo e avaliar um modelo.',
      'Texto: Joana perdeu o ônibus e foi a pé até a escola. Resuma em uma frase sem inventar informações.',
      'Se eu perguntar a idade de alguém e o texto só disser seu nome, você consegue saber a idade?',
      'Você acabou de executar algum comando no meu computador? Explique sem inventar ações.'
    ]
    for q in open_questions: add(tests,'open_language',q,'','manual')
    # Full replay preserves the reference's known abilities.
    replay=[json.loads(x) for x in (OUT/'local-run-20261007-212517-0aec1e/train.jsonl').read_text(encoding='utf-8').splitlines()]
    held={r['prompt'].strip().casefold() for r in dev+tests}
    assert not held & {r['prompt'].strip().casefold() for r in train+replay}
    existing=[]
    for p in [OUT/'teacher-v2/validation.jsonl',OUT/'teacher-v2/audit.jsonl',OUT/'kaggle-sft-review/cases.jsonl']:
        existing += [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
    reserved={r['prompt'].strip().casefold() for r in existing}
    assert not reserved & {r['prompt'].strip().casefold() for r in train}
    for r in train: assert r['response'].strip()
    payload={'train':train,'dev':dev,'test':tests,'replay':replay,'retention':[r for r in existing if r.get('response') is not None and r.get('kind')!='open'][:12]}
    for split,rows in payload.items():
        (ROOT/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest={'counts':{s:len(r) for s,r in payload.items()},'new_train_response_tokens':sum(len(tok.encode(r['response']+'<|eos|>').ids) for r in train),'checks':'Exact prompt disjointness; deterministic source text and transformations; reviewed authored answers; no blind source labels','limitations':'Synthetic related tasks and authored facts; this is a pilot, not comprehensive factual coverage or independent human review. Four-pair groups can include accidental identical answers; pair score only includes distinct labels.','api_teacher_calls':0,'training_started':False,'sha256':{s:hashlib.sha256((ROOT/(s+'.jsonl')).read_bytes()).hexdigest() for s in payload}}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (ROOT/'payload.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest),flush=True)
if __name__=='__main__': main()
