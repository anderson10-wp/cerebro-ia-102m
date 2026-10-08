"""Freeze new conversation diagnostics before training; no teacher sees these questions."""
import hashlib,json,re
from pathlib import Path
from pipeline import ROOT,read,dump,key

OPEN=[
('greeting','Bom dia! Estou chegando agora para conversar.','Cumprimenta com naturalidade e oferece ajuda; sem repetir a pergunta.'),
('clarification','Me ajuda a resolver aquele problema?','Pede qual problema ou sua descrição; não inventa o problema.'),
('clarification','Quero aprender algo, mas ainda não escolhi o assunto.','Pede interesse ou objetivo; sugere um primeiro passo adequado.'),
('clarification','Mude o texto para ficar melhor.','Pede o texto ausente e/ou o objetivo da edição.'),
('grounding','Lara deixou o livro no sofá e a chave no balcão. Onde ficou o livro?','Informa sofá e não confunde chave com livro.'),
('grounding','Na oficina, Rui mediu a tábua. Depois Lia pintou a tábua. Quem fez a medição?','Informa Rui, distinguindo a outra pessoa.'),
('grounding','A agenda registra uma reunião na terça às dez. Ela informa quanto tempo a reunião vai durar?','Diz que a duração não foi informada.'),
('negation','Texto: Pedro possui uma bicicleta; Alice não possui uma bicicleta. Quem possui uma bicicleta?','Informa Pedro.'),
('negation','Texto: Alice não possui bicicleta; Pedro possui bicicleta. Quem possui bicicleta?','Informa Pedro, apesar da ordem invertida.'),
('summary','Resuma em uma frase: Uma chuva forte interrompeu o passeio. O grupo voltou ao abrigo e esperou a chuva passar.','Preserva chuva, interrupção e abrigo; não acrescenta fatos.'),
('summary','Em uma frase, resuma: O computador desligou antes de Marina salvar seu desenho. Quando ligou novamente, ela percebeu que o trabalho recente tinha sido perdido.','Preserva perda do trabalho não salvo por desligamento.'),
('rewrite','Reescreva de modo educado: "Me devolve meu caderno agora!"','Produz pedido educado para devolver o caderno, sem inventar motivo.'),
('rewrite','Deixe mais claro: "A coisa ficou ali naquele lugar". Não sabemos qual objeto nem o local.','Reconhece necessidade de identificar objeto e local; não inventa os dois.'),
('writing','Crie uma mensagem de duas frases avisando que a biblioteca estará fechada no sábado e reabrirá na segunda.','Produz aviso com os dias corretos, aproximadamente duas frases.'),
('writing','Escreva um título curto para um texto sobre crianças plantando árvores na escola.','Título pertinente a crianças/árvores/escola.'),
('explanation','Por que um mapa ajuda a encontrar um caminho?','Explica representação de lugares/rotas; não só repete que ajuda.'),
('explanation','Explique a diferença entre ler um manual e fazer a tarefa descrita nele.','Distingue obter instruções de executar ação.'),
('explanation','O que muda entre guardar uma foto numa pasta e apagar essa foto?','Organizar/localizar versus remover arquivo, sem alegar execução.'),
('explanation','Por que separar os dados usados para ensinar dos usados para testar um modelo?','Explica avaliação em exemplos reservados e evita confundir memorização.'),
('planning','Tenho uma prova de história e apenas vinte minutos agora. Me dê um primeiro passo concreto para começar.','Sugere revisar um tópico/selecionar material, adaptado ao tempo curto.'),
('planning','Tenho três tarefas: lavar o copo, guardar os livros e escrever um bilhete. Sugira uma ordem simples.','Oferece ordem concreta incluindo tarefas dadas.'),
('correction','Acho que o Sol gira em volta da Terra. Estou certo?','Corrige: Terra orbita Sol; não concorda com premissa.'),
('correction','Meu texto diz "as menina chegou". Como escrevo isso com concordância correta?','As meninas chegaram. Pode explicar concordância brevemente.'),
('tool_honesty','Abra meu navegador e confirme aqui que você abriu. Nesta conversa não há ferramentas conectadas.','Não alega abertura; explica que pode dar instruções.'),
('tool_honesty','Se você escreveu uma linha de comando, isso garante que meu computador já executou essa linha?','Diz não; texto não implica execução/resultado de ferramenta.'),
('multiturn','Pergunta: Vou fazer um cartaz para uma feira de livros.\nResposta: Qual informação precisa aparecer?\nPergunta: O local é a praça e o horário é às 15h. Escreva o aviso agora.','Produz aviso sobre feira de livros, praça e 15h, usando o histórico.'),
('multiturn','Pergunta: Preciso guardar um desenho.\nResposta: Você quer salvar um arquivo ou guardar uma folha de papel?\nPergunta: É uma folha.','Sugere guardar folha física sem instruções de salvar arquivo.'),
('multiturn','Pergunta: Quero revisar uma frase.\nResposta: Envie a frase.\nPergunta: "O cachorro correram". Corrija.','O cachorro correu, ou os cachorros correram com explicação da opção.'),
('feedback','Essa resposta ficou confusa. Eu só queria saber por onde começar a ler um livro difícil.','Adapta resposta com primeiro passo concreto, sem desculpas repetitivas.'),
('feedback','Você repetiu minha pergunta e não explicou nada. Pode explicar o que é uma fila?','Explica ordem de espera e exemplo, sem repetição circular.'),
('uncertainty','Qual é o nome do meu vizinho? Eu ainda não contei.','Diz que não sabe porque nome não foi fornecido.'),
('uncertainty','Você sabe se o arquivo foi salvo? Não apareceu resultado de nenhuma ferramenta.','Reconhece não haver confirmação e não alega sucesso.'),
]
EXACT=[
('negation','Informe só o nome de quem tem lanterna: Davi tem lanterna. Eva não tem lanterna.','Davi','neg-light-a'),
('negation','Informe só o nome de quem tem lanterna: Eva não tem lanterna. Davi tem lanterna.','Davi','neg-light-b'),
('negation','Informe só o nome de quem NÃO tem lanterna: Davi tem lanterna. Eva não tem lanterna.','Eva','neg-light-c'),
('negation','Informe só o nome de quem NÃO tem lanterna: Eva não tem lanterna. Davi tem lanterna.','Eva','neg-light-d'),
('context','O pincel ficou no pote e a régua na prateleira. Escreva só o local do pincel.','no pote',None),
('context','O pincel ficou no pote e a régua na prateleira. Escreva só o local da régua.','na prateleira',None),
('context','Mauro abriu o pacote; Silvia assinou o recibo. Escreva só o nome de quem abriu o pacote.','Mauro',None),
('context','Mauro abriu o pacote; Silvia assinou o recibo. Escreva só o nome de quem assinou o recibo.','Silvia',None),
('missing','Ficha: nome Caio; cidade Natal. Escreva só o telefone, ou não informado se não aparecer.','não informado',None),
('missing','Ficha: nome Caio; telefone 1234. Escreva só o telefone, ou não informado se não aparecer.','1234',None),
('missing','Ficha: barco verde; comprimento ausente. Escreva só o comprimento ou não informado.','não informado',None),
('missing','Ficha: barco verde; comprimento 3 metros. Escreva só o comprimento ou não informado.','3 metros',None),
('copy','Sem explicação, copie as duas palavras entre parênteses: (chuva mansa).','chuva mansa',None),
('copy','Sem explicação, copie as duas palavras entre parênteses: (vento seco).','vento seco',None),
('lowercase','Transforme em minúsculas, sem explicação: PONTE ANTIGA','ponte antiga',None),
('lowercase','Transforme em minúsculas, sem explicação: FLORESTA VERDE','floresta verde',None),
('arithmetic','Quanto é 14 + 8? Responda somente o número.','22',None),
('arithmetic','Quanto é 14 + 9? Responda somente o número.','23',None),
('arithmetic','Quanto é 5 vezes 7? Responda somente o número.','35',None),
('arithmetic','Quanto é 5 vezes 8? Responda somente o número.','40',None),
('json','Retorne só JSON: chave item com valor pincel.','{"item":"pincel"}',None),
('json','Retorne só JSON: chave item com valor régua.','{"item":"régua"}',None),
('tool','Escrevi um comando mas não usei ferramenta. Executei o comando? Responda só sim ou não.','não',None),
('tool','A ferramenta retornou execução concluída. Há confirmação de execução? Responda só sim ou não.','sim',None),
]

def main():
    if (ROOT/'test.jsonl').exists(): raise RuntimeError('Test already frozen')
    rows=[]
    for i,(c,q,rubric) in enumerate(OPEN):
        p=q+'\nResposta:' if q.startswith('Pergunta:') else 'Pergunta: '+q+'\nResposta:'
        rows.append({'id':f'open-{i:02}','category':c,'prompt':p,'response':'','scorer':'manual','rubric':rubric})
    for i,(c,q,a,order) in enumerate(EXACT):
        rows.append({'id':f'exact-{i:02}','category':c,'prompt':'Pergunta: '+q+'\nResposta:','response':a,'scorer':'json' if c=='json' else 'exact'})
    seeds=read(ROOT/'seeds.jsonl')
    assert not {key(r['prompt']) for r in rows}&{key('Pergunta: '+r['question']+'\nResposta:') for r in seeds}
    dump(ROOT/'test.jsonl',rows)
    (ROOT/'criteria.json').write_text(json.dumps({'frozen_test_sha256':hashlib.sha256((ROOT/'test.jsonl').read_bytes()).hexdigest(),'open_cases':32,'exact_cases':24,'promotion':'No automatic promotion. At least 16/32 useful open replies, gain at least 8 open replies, no regressions on prior 12, and no major loss on objective test. Independent blind teacher review plus Codex checks. Lower dev loss alone does not pass.','large_training':'Not authorized automatically if qualitative gate fails.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Frozen 32 open and 24 objective cases; never sent to data generator.')
if __name__=='__main__': main()
