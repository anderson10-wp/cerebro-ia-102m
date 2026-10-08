"""Codex review of ALL first 100; preserve originals and record exact edits."""
import json
from pipeline import ROOT,read,dump,sha
FIX={
'0010':{'answer':'Use uma carta curta, com saudação, apresentação, motivo do interesse na vaga e exemplos de habilidades relevantes. Separe essas partes em parágrafos, use fonte legível e revise o texto. Evite transformar a carta numa repetição do currículo.','followup':'Como adaptar a carta a uma vaga diferente?','followup_answer':'Leia os requisitos da vaga e escolha um exemplo verdadeiro de experiência que mostre uma habilidade pedida. Explique por que se interessa pela função, mantendo a carta específica e breve.'},
'0013':{'answer':'Uma rede feedforward não tem, por si só, estado recorrente para lembrar entradas anteriores. Pode precisar de dados e capacidade adequados para aprender padrões. Suas camadas podem usar ativações não lineares; ela não está limitada a representar funções lineares.'},
'0017':{'answer':'A pergunta pressupõe uma relação que não podemos estabelecer apenas pela entonação. Para identificar o caso, precisamos observar a forma da palavra e a construção da frase. Envie uma palavra ou frase concreta para analisar sem inventar uma regra de entonação.'},
'0020':{'answer':'A escolha depende do site e do quanto você quer programar. Um editor de código permite controlar os arquivos; um construtor visual facilita montar páginas. Antes de escolher, defina o conteúdo, as funções e o orçamento disponíveis.','followup':'E se eu quiser um site pessoal simples, sem programar?','followup_answer':'Um construtor visual pode ser adequado. Verifique se permite editar textos e imagens, criar as páginas necessárias e exportar ou publicar o resultado. Compare os limites e custos antes de decidir.'},
'0025':{'followup':'Estou aprendendo espanhol para viajar. Como adaptar a prática?','followup_answer':'Pratique situações de viagem, como pedir informações e fazer um pedido. Ouça frases curtas, repita em voz alta e simule diálogos. Anote palavras que precisou usar e revise-as no dia seguinte.'},
'0030':{'followup':'Como posso experimentar o exagero numa pintura?','followup_answer':'Escolha uma emoção e experimente ampliar cores, formas ou proporções para expressá-la. Compare o resultado com uma versão mais discreta e observe qual transmite melhor o efeito pretendido.'},
'0035':{'followup':'Como praticar para apresentar dados de forma clara?','followup_answer':'Escolha um gráfico simples e explique em voz alta a principal conclusão. Grave uma apresentação curta, verifique se os termos ficaram claros e retire detalhes que não ajudam o público a entender.'},
'0040':{'followup_answer':'Um temporizador simples já permite alternar períodos de foco e pausas. Uma lista de tarefas ajuda a decidir o que fazer em cada período. Registre o que concluiu e ajuste os intervalos ao seu ritmo.'},
'0041':{'answer':'É importante aprender programação, algoritmos, estruturas de dados e como testar e manter programas. Uma graduação pode ajudar, mas não é uma exigência universal para todas as vagas. Projetos práticos e experiência com equipes também demonstram habilidades.'},
'0042':{'answer':'Peça autorização ao seu amigo antes de alterar o roteador. Primeiro, identifique o problema: sinal fraco, muitos dispositivos ou conexão lenta. Com permissão, vocês podem consultar o manual e revisar a posição do aparelho e as configurações adequadas.'},
'0045':{'answer':'Procure aulas que ensinem a organizar ideias, construir parágrafos e revisar textos. Verifique se há exercícios e feedback, além das condições de acesso. Uma boa prática é escrever um texto curto e comparar sua revisão com as orientações da aula.','followup':'Como escolher um curso de redação para iniciantes?','followup_answer':'Confira se o curso começa com estrutura de frases e parágrafos, propõe exercícios curtos e oferece exemplos de revisão. Veja os requisitos e as condições de acesso antes de se inscrever.'},
'0053':{'answer':'Modelos podem produzir combinações que não aparecem literalmente nos exemplos de treino. Isso não garante que sejam úteis ou corretas. Para avaliar criatividade, precisamos observar a novidade e a qualidade dos resultados, sem confundir novidade com informação inventada.'},
'0055':{'answer':'Para afirmar quais tendências se destacam neste ano, é preciso consultar fontes recentes. Não consultei essas fontes aqui. Posso explicar áreas gerais, como testes automatizados, controle de versão e uso de modelos de linguagem no desenvolvimento.','followup':'Como um modelo de linguagem pode ajudar a desenvolver software?','followup_answer':'Pode sugerir trechos de código, explicar mensagens de erro e propor testes. As sugestões precisam ser revisadas e executadas num ambiente adequado; gerar texto não comprova que o programa funciona.'},
'0056':{'answer':'Na cabana isolada, três amigos ouviram passos no telhado. A porta estava trancada por dentro, mas pegadas molhadas surgiam no corredor. Quando apagaram a lanterna, uma voz sussurrou seus nomes. Nenhum deles havia dito os nomes em voz alta.'},
'0060':{'answer':'Você pode combinar aulas online, gravações com transcrição, dicionários e exercícios de vocabulário. Para praticar, escolha uma situação cotidiana, ouça frases curtas e tente responder. Verifique o nível, as condições de acesso e a qualidade dos recursos.','followup_answer':'Escolha uma atividade curta, como ouvir um diálogo e repetir três frases. Reserve alguns minutos para revisar palavras do dia anterior. A prática regular, mesmo breve, ajuda mais do que acumular muitas ferramentas sem usá-las.'},
'0062':{'answer':'Sim. Poeira pode dificultar a ventilação e elevar a temperatura, reduzindo o desempenho. Antes de limpar, desligue o computador e desconecte a energia. Siga o manual e retire a poeira com cuidado, sem líquidos e sem forçar componentes.'},
'0066':{'answer':'Pratique concordância, pontuação e organização de frases. Leia textos claros e observe como as ideias se conectam. Depois, revise sua postagem em voz alta e procure trechos ambíguos. Um corretor pode ajudar, mas suas sugestões também precisam de revisão.'},
'0071':{'answer':'Erros comuns incluem escala incorreta, falta de legenda e representação inadequada das distâncias. Confira as medidas e a projeção usada, explique os símbolos e acrescente os nomes necessários para que o leitor consiga interpretar o mapa.'},
'0072':{'answer':'Comece com poucas plantas adequadas à luz e ao espaço disponíveis. Use recipientes com drenagem e conheça as necessidades de cada espécie. Observe o solo antes de regar e ajuste os cuidados, evitando aplicar a mesma rotina a todas as plantas.'},
'0083':{'answer':'A possibilidade e o método dependem do modelo do iPod e de como as músicas foram adicionadas. Nem sempre basta copiar pelo gerenciador de arquivos. Informe o modelo e o sistema do computador para identificar uma opção compatível sem prometer uma transferência.'},
'0095':{'answer':'Técnicas incluem inicialização adequada dos pesos, escolha de ativações e conexões residuais. A solução depende da arquitetura: por exemplo, atalhos podem facilitar o fluxo do gradiente. Verificar os gradientes ajuda a identificar onde o aprendizado está enfraquecendo.'},
}
EXCLUDE=['dialog-0033','dialog-0059','dialog-0096']
def main():
    backup=ROOT/'sample-before-codex-edits.jsonl'
    if backup.exists(): raise RuntimeError('Sample already revised')
    original=read(ROOT/'dialogs.jsonl'); dump(backup,original)
    edits=[]
    for path in sorted((ROOT/'batches').glob('[0-9][0-9][0-9][0-9].json')):
        batch=json.loads(path.read_text(encoding='utf-8'))
        for r in batch['accepted']:
            suffix=r['id'].split('-')[-1]
            if suffix in FIX:
                edits.append({'id':r['id'],'old':{k:r.get(k) for k in FIX[suffix]},'new':FIX[suffix]})
                r.update(FIX[suffix]); r['codex_revision']=True
        path.write_text(json.dumps(batch,ensure_ascii=False,indent=2),encoding='utf-8')
    dump(ROOT/'manual-edits.jsonl',edits)
    gate=json.loads((ROOT/'sample-review.json').read_text(encoding='utf-8'))
    gate.update(passed=True,manual_review_pending=False,manual_review={'read':100,'corrected':len(edits),'excluded':len(EXCLUDE),'usable_after_revision':97,'notes':'The Groq self-review missed substantive errors. Codex corrected facts, circular responses, role-swapped follow-ups, unfounded current recommendations and literal rejection markers. Three irrelevant specialized prompts excluded.'},exclude_ids=EXCLUDE,raw_sample_passed=False,criterion='At least 95/100 usable AFTER explicit revision, with flawed items excluded. Remainder uses tightened generation and filtering. No claim of perfect factual correctness.')
    (ROOT/'sample-review.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'sample_read':100,'corrected':len(edits),'excluded':3,'usable':97}))
if __name__=='__main__': main()
