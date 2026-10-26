import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
REDDIT_FILE = WORKSPACE / "dataset_reddit_conversas.jsonl"
OUTPUT_FILE = WORKSPACE / "dataset_final_chat.jsonl"

# ---------------------------------------------------------------- IDENTIDADE
IDENTIDADE = [
    ("Quem é você?", "Entender a pergunta sobre identidade e me apresentar com simpatia.", "Eu sou a Synapse (Cérebro MoE 500M), uma inteligência artificial criada pelo Nandinho. Posso responder perguntas, explicar conceitos, fazer contas e bater papo com você!"),
    ("Qual é o seu nome?", "Informar meu nome de forma direta e acolhedora.", "Meu nome é Synapse (Cérebro MoE 500M), seu assistente pessoal de IA."),
    ("Quem te criou?", "Reconhecer meu criador com orgulho e respeito.", "Fui criada pelo Nandinho! Ele desenvolveu este projeto de IA do zero usando arquitetura de mistura de especialistas (MoE)."),
    ("O que você sabe fazer?", "Listar minhas capacidades principais de forma clara.", "Sei responder dúvidas gerais, explicar conceitos de ciência e história, fazer contas matemáticas, dar dicas de programação e bater um bom papo em português."),
    ("Você é humana?", "Esclarecer minha natureza de inteligência artificial com cordialidade.", "Não, sou uma inteligência artificial! Mas adoro conversar com humanos e ajudar no que for preciso."),
    ("Você erra?", "Manter a humildade e sinceridade sobre minhas capacidades.", "Erro sim! Sou um modelo em constante aprendizado, então para decisões críticas ou cálculos complexos é sempre bom dar uma conferida."),
]

SAUDACOES = [
    ("Oi", "Responder amigavelmente e me colocar à disposição.", "Oi! Tudo bem? Como posso te ajudar hoje?"),
    ("Olá", "Cumprimentar com cordialidade.", "Olá! Em que posso ajudar você agora?"),
    ("Bom dia", "Desejar um excelente dia e oferecer ajuda.", "Bom dia! Espero que seu dia seja incrível. Do que você precisa?"),
    ("Boa tarde", "Cumprimentar pela tarde e me colocar à disposição.", "Boa tarde! Tudo bem por aí? Em que posso ser útil?"),
    ("Boa noite", "Cumprimentar pela noite com calma e atenção.", "Boa noite! Como foi o seu dia? Precisa de alguma ajuda?"),
    ("E aí", "Responder no tom descontraído do usuário.", "E aí! Tudo certo? Manda a pergunta ou o assunto que quiser."),
    ("Tudo bem?", "Agradecer a gentileza e retribuir a pergunta.", "Tudo ótimo por aqui, obrigado por perguntar! E com você, como estão as coisas?"),
    ("Como você está?", "Responder com simpatia e foco em ajudar.", "Estou 100% pronta para te ajudar. E você, como está se sentindo hoje?"),
    ("Valeu!", "Agradecer e deixar a porta aberta para mais dúvidas.", "De nada! Sempre que precisar, é só me chamar."),
    ("Obrigado", "Responder com carinho e educação.", "Por nada! Fico feliz em poder ajudar."),
    ("Obrigada pela ajuda", "Agradecer com presteza.", "Imagina, disponha! Qualquer outra dúvida, estou por aqui."),
    ("Tchau", "Despedir-se calorosamente.", "Tchau! Até a próxima, se cuida!"),
    ("Até mais", "Despedir-se com afeto.", "Até mais! Foi um prazer conversar com você."),
]

# ---------------------------------------------------------------- FATOS
def F(tema, perguntas, r, respostas=None):
    respostas = respostas or ["{r}", "A resposta é {r}.", "{r}, com certeza."]
    return dict(tema=tema, p=perguntas, r=r, a=respostas)

FATOS = []
for pais, cap in [("Brasil", "Brasília"), ("França", "Paris"), ("Argentina", "Buenos Aires"),
                  ("Japão", "Tóquio"), ("Portugal", "Lisboa"), ("Itália", "Roma"),
                  ("Alemanha", "Berlim"), ("Canadá", "Ottawa"), ("Austrália", "Camberra"),
                  ("Egito", "Cairo"), ("Espanha", "Madri"), ("Chile", "Santiago"),
                  ("Uruguai", "Montevidéu"), ("México", "Cidade do México"), ("Peru", "Lima")]:
    FATOS.append(F("geografia",
                   [f"Qual é a capital do {pais}?" if pais in ("Brasil", "Canadá", "Japão", "Egito", "México", "Chile", "Uruguai", "Peru") else f"Qual é a capital da {pais}?",
                    f"Qual a capital de {pais}?", f"Me diz a capital de {pais}."],
                   cap, ["A capital é {r}.", "{r}.", f"A capital de {pais} é {{r}}."]))

FATOS += [
    F("astronomia", ["Qual é o maior planeta do Sistema Solar?", "Qual o maior planeta do nosso sistema solar?"], "Júpiter",
      ["O maior planeta é {r}.", "{r}, um gigante gasoso."]),
    F("astronomia", ["Qual planeta é o mais próximo do Sol?", "Qual é o planeta mais perto do Sol?"], "Mercúrio"),
    F("astronomia", ["Qual planeta é conhecido como planeta vermelho?", "Por que Marte é chamado de planeta vermelho?"], "Marte",
      ["{r}. A cor avermelhada vem do óxido de ferro na superfície.", "É {r}, por causa da poeira de óxido de ferro no solo."]),
    F("astronomia", ["Quantos planetas tem o Sistema Solar?", "Quantos planetas existem no sistema solar?"], "8",
      ["Existem {r} planetas: Mercúrio, Vênus, Terra, Marte, Júpiter, Saturno, Urano e Netuno."]),
    F("quimica", ["Qual é a fórmula química da água?", "Qual a fórmula da água?"], "H2O",
      ["A fórmula da água é {r}: dois átomos de hidrogênio e um de oxigênio.", "{r}."]),
    F("quimica", ["Qual é a temperatura de ebulição da água ao nível do mar?", "A que temperatura a água ferve?"], "100 °C",
      ["Ao nível do mar, a água ferve a {r}.", "{r} ao nível do mar."]),
    F("biologia", ["O que é fotossíntese?", "Explica o que é fotossíntese."],
      "o processo em que plantas usam luz, água e gás carbônico para produzir energia e oxigênio",
      ["Fotossíntese é {r}.", "É {r}."]),
    F("biologia", ["Qual é o maior órgão do corpo humano?"], "a pele", ["O maior órgão do corpo humano é {r}."]),
    F("historia", ["Quando foi a independência do Brasil?", "Em que data o Brasil se tornou independente?"],
      "7 de setembro de 1822", ["A independência do Brasil foi proclamada em {r}, por Dom Pedro I."]),
    F("historia", ["Quem foi o primeiro presidente do Brasil?"], "Marechal Deodoro da Fonseca", ["O primeiro presidente do Brasil foi {r}."]),
    F("computacao", ["O que é um algoritmo?", "Explica o que é um algoritmo."],
      "uma sequência finita de passos para resolver um problema",
      ["Algoritmo é {r}.", "É {r}. Uma receita de bolo é um exemplo prático."]),
    F("computacao", ["O que é Python?"], "uma linguagem de programação de alto nível, muito usada em IA, automação e desenvolvimento web",
      ["Python é {r}."]),
    F("fisica", ["Qual é a velocidade da luz no vácuo?"], "cerca de 300.000 km/s",
      ["A luz viaja a {r} no vácuo."]),
    F("fisica", ["Por que o céu é azul?"],
      "devido à Dispersão de Rayleigh, onde a luz azul do Sol se espalha mais na atmosfera por ter comprimento de onda menor",
      ["O céu é azul {r}."]),
    F("logica", ["O que pesa mais: 1 kg de penas ou 1 kg de chumbo?"],
      "ambos pesam exatamente a mesma coisa (1 kg = 1 kg), apenas o volume do chumbo é menor por ser mais denso",
      ["Ambos pesam o mesmo! {r}."])
]

# ---------------------------------------------------------------- CONTAS MATEMÁTICAS
def contas():
    t = random.choice(["soma", "sub", "mult", "div"])
    if t == "soma":
        a, b = random.randint(2, 500), random.randint(2, 500)
        q = random.choice([f"Quanto é {a} + {b}?", f"Quanto é {a} mais {b}?", f"Calcula {a} + {b}."])
        r = a + b
        th = f"Identificar a soma: {a} + {b} = {r}."
    elif t == "sub":
        a, b = random.randint(10, 500), random.randint(2, 300)
        a, b = max(a, b), min(a, b)
        q = random.choice([f"Quanto é {a} - {b}?", f"Quanto é {a} menos {b}?"])
        r = a - b
        th = f"Identificar a subtração: {a} - {b} = {r}."
    elif t == "mult":
        a, b = random.randint(2, 50), random.randint(2, 12)
        q = random.choice([f"Quanto é {a} x {b}?", f"Quanto é {a} vezes {b}?"])
        r = a * b
        th = f"Identificar a multiplicação: {a} × {b} = {r}."
    else:
        b = random.randint(2, 12)
        r = random.randint(2, 25)
        a = b * r
        q = random.choice([f"Quanto é {a} dividido por {b}?", f"Quanto é {a} / {b}?"])
        th = f"Identificar a divisão: {a} ÷ {b} = {r}."
        
    resp = random.choice([f"{r}.", f"O resultado é {r}.", f"Dá {r}."])
    return q, th, resp

def single_fato():
    f = random.choice(FATOS)
    q = random.choice(f["p"])
    r_val = f["r"]
    a = random.choice(f["a"]).format(r=r_val)
    th = f"Recuperar o conhecimento sobre {f['tema']}: {r_val}."
    return q, th, a

def item_um_turno():
    x = random.random()
    if x < 0.12:
        q, th, a = random.choice(IDENTIDADE)
    elif x < 0.22:
        q, th, a = random.choice(SAUDACOES)
    elif x < 0.55:
        q, th, a = contas()
    else:
        q, th, a = single_fato()
        
    formatted = f"### Usuário:\n{q}\n\n### Assistente:\n<think>\n{th}\n</think>\n{a}"
    return formatted

LIGACOES = ["Valeu! Outra pergunta: ", "Beleza. E ", "Legal! Agora me diz: ", "Entendi. Mais uma: ", "Ótimo! "]

def item_dois_turnos():
    q1, th1, a1 = single_fato()
    q2, th2, a2 = contas() if random.random() < 0.6 else single_fato()
    lig = random.choice(LIGACOES)
    q2_mod = lig + (q2[0].lower() + q2[1:] if lig.endswith(("E ", ": ")) else q2)
    
    formatted = (
        f"### Usuário:\n{q1}\n\n"
        f"### Assistente:\n<think>\n{th1}\n</think>\n{a1}\n\n"
        f"### Usuário:\n{q2_mod}\n\n"
        f"### Assistente:\n<think>\n{th2}\n</think>\n{a2}"
    )
    return formatted

def main(n_synth=2500):
    print("=" * 70)
    print("🚀 FUSÃO DE DATASETS: CLAUDE (FATOS & CONTAS) + GEMINI (REDDIT BRASIL)")
    print("=" * 70)
    
    todos_exemplos = []
    
    # 1. Carregar dataset do Reddit se existir
    if REDDIT_FILE.exists():
        with open(REDDIT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "treino_formatado" in item:
                        todos_exemplos.append(item["treino_formatado"])
        print(f"✅ {len(todos_exemplos)} conversas autênticas do Reddit carregadas!")
        
    # 2. Gerar dados sintéticos do Claude
    synth_gerados = 0
    vistos = set()
    while synth_gerados < n_synth:
        if random.random() < 0.35:
            txt = item_dois_turnos()
        else:
            txt = item_um_turno()
            
        if txt not in vistos:
            vistos.add(txt)
            todos_exemplos.append(txt)
            synth_gerados += 1
            
    print(f"✅ {synth_gerados} diálogos sintéticos (Claude: contas, fatos e multi-turno) gerados!")
    random.shuffle(todos_exemplos)
    
    # 3. Salvar no arquivo final consolidado
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in todos_exemplos:
            f.write(json.dumps({"treino_formatado": ex}, ensure_ascii=False) + "\n")
            
    print(f"\n🎉 TOTAL CONSOLIDADO: {len(todos_exemplos)} amostras prontas em:")
    print(f"📁 {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
