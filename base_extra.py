"""Base de dados complementares para o gerador v2 da Synapse (Cérebro MoE)."""

def F(tema, perguntas, r, respostas=None):
    respostas = respostas or ["{r}", "A resposta é {r}.", "{r}, com certeza."]
    return dict(tema=tema, p=perguntas, r=r, a=respostas)

FATOS_EXTRA = [
    F("geografia", ["Qual é a maior floresta tropical do mundo?"], "a Floresta Amazônica", ["A maior floresta tropical do mundo é {r}."]),
    F("geografia", ["Onde fica o Cristo Redentor?"], "no Rio de Janeiro, no topo do morro do Corcovado"),
    F("historia", ["Quem foi Zumbi dos Palmares?"], "um dos maiores líderes quilombolas do Brasil, símbolo da resistência contra a escravidão"),
    F("historia", ["Quem foi Santos Dumont?"], "um pioneiro da aviação brasileira que voou com o 14-Bis em Paris"),
    F("fisica", ["Quem formulou a famosa equação E=mc²?"], "Albert Einstein"),
    F("computacao", ["O que é HTML?"], "a linguagem padrão para estruturar páginas na web"),
    F("computacao", ["O que significa a sigla IA?"], "Inteligência Artificial"),
    F("biologia", ["Qual é o animal mais rápido do mundo em terra?"], "o guepardo"),
    F("biologia", ["As baleias são peixes ou mamíferos?"], "mamíferos", ["As baleias são {r}."]),
    F("culinaria", ["De onde é típico o pão de queijo?"], "de Minas Gerais", ["O pão de queijo é uma receita típica {r}."]),
    F("culinaria", ["Qual é o prato mais famoso da Bahia?"], "o acarajé", ["Um dos pratos mais emblemáticos da Bahia é {r}."]),
    F("logica", ["Se você tem 3 maçãs e come 1, com quantas maçãs fica?"], "2 maçãs", ["Você fica com {r}, porque 3 - 1 = 2."]),
]

NOMES = [
    "Nandinho", "Lucas", "Mariana", "Gabriel", "Beatriz", "Carlos", "Juliana",
    "Rodrigo", "Camila", "Felipe", "Larissa", "Rafael", "Ana", "Pedro", "Bruno",
    "Fernanda", "Eduardo", "Amanda", "Gustavo", "Letícia"
]

APRESENTACOES = [
    ("Meu nome é {n}.", "Prazer em te conhecer, {n}! Como posso te ajudar hoje?"),
    ("Oi, pode me chamar de {n}.", "Olá, {n}! Tudo bem com você? Em que posso ser útil?"),
    ("Eu sou o {n}.", "Fala, {n}! Tudo certo? O que manda hoje?"),
    ("Meu nome é {n}, prazer.", "Prazer, {n}! Estou aqui para ajudar no que precisar."),
]

PERGUNTAS_NOME = [
    ("Você lembra o meu nome?", "Lembro sim! Seu nome é {n}."),
    ("Qual é o meu nome mesmo?", "Seu nome é {n}."),
    ("Você sabe qual é o meu nome?", "Sei sim, você é o {n}!"),
    ("Como eu me chamo?", "Você me disse que se chama {n}."),
]

CASUAL = [
    (["O que você gosta de fazer?", "Você tem algum passatempo?"], 
     "Como uma IA, eu gosto de processar textos, aprender coisas novas e bater papo com pessoas legais como você!"),
    (["Conta uma piada", "Me conta uma piada?", "Sabe alguma piada?"], 
     "Por que o livro de matemática se suicidou? Porque tinha muitos problemas!"),
    (["Você dorme?", "IA dorme?", "Você descansa?"], 
     "Eu não durmo! Fico acordada 24 horas por dia, pronta para quando você precisar tirar alguma dúvida ou conversar."),
    (["O que você acha do Brasil?"], 
     "O Brasil é um país fascinante, com uma cultura riquíssima, povo caloroso, culinária incrível e uma biodiversidade sem igual!"),
    (["Está calor hoje, né?", "Que calor!"], 
     "Pois é! Em dias quentes o melhor remédio é se hidratar bem, tomar uma água gelada e ficar na sombra."),
]

NAO_SEI = [
    (["Qual é o sentido da vida?", "Qual o propósito da vida humana?"], 
     "Essa é uma das maiores perguntas da humanidade! Filósofos discutem isso há milênios. Muitos acreditam que o sentido é encontrar felicidade, aprender e cuidar de quem a gente ama."),
    (["Quem vai ganhar o campeonato amanhã?", "Quem vai ser o campeão?"], 
     "Eu não consigo prever o futuro! Vai depender do desempenho dos times em campo, mas que vença o melhor!"),
    (["Quantos grãos de areia existem no mundo todo?"], 
     "Não é possível saber com precisão exata, mas cientistas estimam que haja algo em torno de 7 quintilhões e meio de grãos de areia nas praias e desertos da Terra!"),
]

CODIGO = [
    (["Como fazer um Hello World em Python?", "Escreve um hello world em python"], 
     "É super simples! Basta escrever:\n```python\nprint('Olá, Mundo!')\n```"),
    (["Como somar dois números em Python?"], 
     "Você pode fazer assim:\n```python\ndef somar(a, b):\n    return a + b\n\nprint(somar(10, 20)) # Imprime 30\n```"),
    (["Como fazer um loop de 1 a 5 em Python?"], 
     "Você usa o comando `for` com `range`:\n```python\nfor i in range(1, 6):\n    print(i)\n```"),
    (["Como ler um arquivo de texto em Python?"], 
     "A forma mais segura é usando o bloco `with open`:\n```python\nwith open('arquivo.txt', 'r', encoding='utf-8') as f:\n    conteudo = f.read()\n    print(conteudo)\n```"),
]

DIALOGOS = [
    ("Oi, tudo bem?", "Olá! Tudo ótimo por aqui, e com você?", 
     "Tudo bem também. Qual a capital do Japão?", "A capital do Japão é Tóquio!"),
    ("Bom dia!", "Bom dia! Como posso ajudar você hoje?", 
     "Quanto é 15 x 3?", "15 vezes 3 é igual a 45."),
    ("Você sabe programar?", "Sei sim! Posso te ajudar com linguagens como Python, JavaScript e conceitos de lógica de programação.", 
     "Me dá uma dica de Python?", "Uma ótima dica em Python é usar list comprehensions para deixar o código limpo e elegante! Por exemplo: `quadrados = [x**2 for x in range(10)]`."),
]
