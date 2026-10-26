"""Gerador v2 de dataset de conversas PT-BR para a Synapse.

Uso: python gerar_dataset_v2.py [N] [saida_completa] [saida_ouro]
Gera dois arquivos JSONL:
  dataset_ouro.jsonl : só conteúdo escrito à mão (sem variações aleatórias)
  dataset_v2.jsonl   : ouro + variações de pergunta + contas + diálogos de memória
                       + pares de perguntas do mesmo assunto (limitado por cotas)
Cada linha: {"tema", "origem", "messages": [{"role", "content"}, ...]}
"""
import json
import random
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

import gerar_dataset as base
from gerar_dataset import FATOS, IDENTIDADE, SAUDACOES
from base_extra import (FATOS_EXTRA, DIALOGOS, CASUAL, CODIGO, NAO_SEI, NOMES,
                        APRESENTACOES, PERGUNTAS_NOME)

random.seed(7)

# perguntas da base antiga cujas respostas ficam estranhas ou que foram refeitas
REMOVER_Q = {"O que é uma célula?", "Por que Marte é chamado de planeta vermelho?"}
INICIOS_RUINS = {"para", "há", "pelo", "organizar", "cerca", "por", "ao", "de", "em"}


def cap1(s):
    s = s.strip()
    return (s[0].upper() + s[1:]) if s and s[0].isalpha() else s


def ajustar(a):
    a = cap1(a)
    if len(a.split()) > 3 and a[-1].isalnum():
        a += "."
    while a.endswith(".."):
        a = a[:-1]
    return a


def template_ok(t, r):
    """Evita 'A resposta é para fazer...' em definições longas e o 'sem dúvida' (confiança demais)."""
    if t.endswith("sem dúvida."):
        return False
    if t.startswith("A resposta é"):
        return r.split()[0].lower() not in INICIOS_RUINS and len(r.split()) <= 6
    return True


def expandir_fato(f):
    """Todas as combinações pergunta x resposta de um fato."""
    saida = []
    for q in f["p"]:
        if q in REMOVER_Q:
            continue
        tema = "capitais" if "capital" in q.lower() else f["tema"]
        for t in f["a"]:
            if not template_ok(t, f["r"]):
                continue
            saida.append((tema, q, ajustar(t.format(r=f["r"]))))
    return saida


PREFIXOS = ["", "", "", "Uma dúvida: ", "Me diz uma coisa: ", "Pode me responder? ",
            "Rapidinho: ", "Tenho uma pergunta: ", "Ei, "]


def variar(q):
    p = random.choice(PREFIXOS)
    if p:
        q = p + q[0].lower() + q[1:]
    if q.endswith("?") and random.random() < 0.15:
        q = q[:-1] + ", por favor?"
    return q


# ------------------------------------------------------------ contas e problemas
def contas2():
    t = random.choice(["pct", "troco", "pacotes", "horas", "quadrado", "raiz", "seq", "parimpar"])
    if t == "pct":
        p = random.choice([5, 10, 20, 25, 50])
        n = random.choice(range(20, 1001, 20))
        r = n * p // 100
        return "matematica", random.choice([f"Quanto é {p}% de {n}?", f"Calcula {p}% de {n}."]), \
            random.choice([f"{p}% de {n} é {r}.", f"Dá {r}."])
    if t == "troco":
        nome = random.choice(["Maria", "João", "Ana", "Pedro", "Lucas", "Carla"])
        a = random.randint(20, 500)
        b = random.randint(5, a - 1)
        return "matematica", f"{nome} tinha {a} reais e gastou {b}. Com quanto ficou?", \
            f"{nome} ficou com {a - b} reais, porque {a} - {b} = {a - b}."
    if t == "pacotes":
        a, b = random.randint(3, 30), random.randint(2, 12)
        return "matematica", f"Um pacote tem {a} biscoitos. Quantos biscoitos há em {b} pacotes?", \
            f"Em {b} pacotes há {a * b} biscoitos, porque {a} x {b} = {a * b}."
    if t == "horas":
        h = random.randint(2, 12)
        return "matematica", f"Quantos minutos tem {h} horas?", f"{h} horas têm {h * 60} minutos."
    if t == "quadrado":
        a = random.randint(2, 30)
        return "matematica", f"Quanto é {a} ao quadrado?", f"{a} ao quadrado é {a * a}."
    if t == "raiz":
        a = random.randint(2, 30)
        return "matematica", f"Qual é a raiz quadrada de {a * a}?", f"A raiz quadrada de {a * a} é {a}."
    if t == "seq":
        s, d = random.randint(1, 20), random.randint(2, 9)
        seq = ", ".join(str(s + i * d) for i in range(4))
        return "matematica", f"Qual é o próximo número da sequência {seq}?", \
            f"O próximo é {s + 4 * d}, porque a sequência soma {d} a cada termo."
    n = random.randint(2, 999)
    return "matematica", f"O número {n} é par ou ímpar?", f"{n} é {'par' if n % 2 == 0 else 'ímpar'}."


def contas_todas():
    if random.random() < 0.5:
        t, q, a = base.contas()
        return t, q, a
    return contas2()


# ------------------------------------------------------------------ montagem
def msgs(*pares):
    out = []
    for q, a in pares:
        out += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    saida = sys.argv[2] if len(sys.argv) > 2 else "dataset_v2.jsonl"
    saida_ouro = sys.argv[3] if len(sys.argv) > 3 else "dataset_ouro.jsonl"

    vistos, ouro, extra = set(), [], []

    def add(lista, tema, origem, mensagens):
        chave = tuple(m["content"] for m in mensagens)
        if chave in vistos:
            return False
        vistos.add(chave)
        lista.append({"tema": tema, "origem": origem, "messages": mensagens})
        return True

    # --- 1) OURO: tudo escrito à mão, uma vez cada
    fatos = []
    for f in FATOS + FATOS_EXTRA:
        fatos += expandir_fato(f)
    TABELAS = {"capitais", "moedas"}   # geradas de tabela: ficam fora do ouro e com cota
    for t, q, a in fatos:
        if t not in TABELAS:
            add(ouro, t, "manual", msgs((q, a)))
    for qs, a in CASUAL:
        for q in qs:
            add(ouro, "casual", "manual", msgs((q, a)))
    for qs, a in NAO_SEI:
        for q in qs:
            add(ouro, "honestidade", "manual", msgs((q, a)))
    for qs, a in CODIGO:
        for q in qs:
            add(ouro, "codigo", "manual", msgs((q, a)))
    for q, a in IDENTIDADE:
        add(ouro, "identidade", "manual", msgs((q, a)))
    for q, a in SAUDACOES:
        add(ouro, "saudacao", "manual", msgs((q, a)))
    for q1, a1, q2, a2 in DIALOGOS:
        add(ouro, "dialogo", "manual", msgs((q1, a1), (q2, a2)))

    # --- 2) EXTRA: variações e conteúdo gerado por template
    nao_cap = [x for x in fatos if x[0] not in TABELAS]
    tabela = [x for x in fatos if x[0] in TABELAS]
    random.shuffle(tabela)

    # variações de pergunta (prefixos) nos fatos escritos à mão
    for t, q, a in nao_cap:
        for _ in range(4):
            add(extra, t, "manual_variacao", msgs((variar(q), a)))
    # capitais e moedas (tabela): amostra limitada para não dominar o dataset
    for t, q, a in tabela[:350]:
        add(extra, t, "tabela", msgs((q, a)))
    for qs, a in CASUAL + NAO_SEI + CODIGO:
        for q in qs:
            for _ in range(2):
                add(extra, "variacao", "manual_variacao", msgs((variar(q), a)))

    # diálogos de memória do nome
    for nome in NOMES:
        for (qa, aa) in APRESENTACOES:
            for (qb, ab) in PERGUNTAS_NOME:
                add(extra, "memoria_nome", "template",
                    msgs((qa.format(n=nome), aa.format(n=nome)), (qb, ab.format(n=nome))))

    # contas e problemas (cota)
    cota_math = min(int(n * 0.22), 700)
    tent = 0
    cont = 0
    while cont < cota_math and tent < cota_math * 30:
        tent += 1
        t, q, a = contas_todas()
        cont += add(extra, t, "template", msgs((q, a)))

    # pares de perguntas do MESMO assunto (cota)
    por_tema = {}
    for t, q, a in nao_cap:
        por_tema.setdefault(t, []).append((q, a))
    liga = ["E ", "Agora, ", "Beleza! ", "Valeu! Outra dúvida: "]
    cota_comb = min(int(n * 0.12), 900)
    tent = 0
    cont = 0
    temas = [t for t, v in por_tema.items() if len(v) >= 2]
    while cont < cota_comb and tent < cota_comb * 30:
        tent += 1
        t = random.choice(temas)
        (q1, a1), (q2, a2) = random.sample(por_tema[t], 2)
        l = random.choice(liga)
        q2b = l + (q2 if l.startswith("Beleza") else q2[0].lower() + q2[1:])
        cont += add(extra, t, "template", msgs((q1, a1), (q2b, a2)))

    random.shuffle(extra)
    total = ouro + extra
    if len(total) > n:
        total = ouro + extra[: max(0, n - len(ouro))]
    random.shuffle(total)

    for caminho, linhas in ((saida_ouro, ouro), (saida, total)):
        with open(caminho, "w", encoding="utf-8") as fh:
            for l in linhas:
                fh.write(json.dumps(l, ensure_ascii=False) + "\n")

    print(f"ouro: {len(ouro)} -> {saida_ouro}")
    print(f"completo: {len(total)} -> {saida}")
    print("por origem:", dict(Counter(l["origem"] for l in total)))
    print("por tema:", Counter(l["tema"] for l in total).most_common())


if __name__ == "__main__":
    main()
