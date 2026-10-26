import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
V2_FILE = WORKSPACE / "dataset_v2.jsonl"
REDDIT_FILE = WORKSPACE / "dataset_reddit_conversas.jsonl"
FINAL_FILE = WORKSPACE / "dataset_final_chat.jsonl"

def gerar_pensamento(content, tema):
    c_low = content.lower()
    if any(k in c_low for k in ["+", "-", "x", "*", "dividido", "/", "%", "raiz", "quadrado", "reais", "gastou", "ficou", "biscoitos"]):
        return "Analisar os valores da questão matemática e realizar o cálculo com precisão passo a passo."
    elif any(k in c_low for k in ["capital", "planeta", "fórmula", "graus", "ano", "quem", "quando", "onde"]):
        return "Identificar a pergunta factual e resgatar a informação correta na memória de conhecimentos."
    elif any(k in c_low for k in ["oi", "olá", "bom dia", "boa tarde", "boa noite", "e aí", "tudo bem"]):
        return "Identificar a saudação do usuário e responder com cordialidade e simpatia."
    elif any(k in c_low for k in ["nome", "chamo", "lembra"]):
        return "Verificar o contexto anterior para lembrar e confirmar o nome do usuário com atenção."
    elif any(k in c_low for k in ["python", "código", "programar", "função", "loop"]):
        return "Entender o pedido técnico de programação e fornecer um exemplo limpo, didático e funcional."
    else:
        return "Compreender a mensagem do usuário e formular uma resposta amigável, clara e prestativa."

amostras_finais = []

# 1. Carregar conversas do Reddit Brasil (303 amostras ricas)
if REDDIT_FILE.exists():
    with open(REDDIT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if "treino_formatado" in item:
                    amostras_finais.append(item["treino_formatado"])
    print(f"✅ Reddit Brasil: {len(amostras_finais)} conversas autênticas carregadas.")

# 2. Carregar e converter dataset V2 do Claude (2.192 amostras)
if V2_FILE.exists():
    count_v2 = 0
    with open(V2_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            msgs = item.get("messages", [])
            if not msgs:
                continue
                
            tema = item.get("tema", "geral")
            dialogo_formatado = []
            
            for i in range(0, len(msgs), 2):
                if i + 1 < len(msgs):
                    u_msg = msgs[i]["content"].strip()
                    a_msg = msgs[i+1]["content"].strip()
                    th = gerar_pensamento(u_msg, tema)
                    bloco = f"### Usuário:\n{u_msg}\n\n### Assistente:\n<think>\n{th}\n</think>\n{a_msg}"
                    dialogo_formatado.append(bloco)
                    
            if dialogo_formatado:
                texto_completo = "\n\n".join(dialogo_formatado)
                amostras_finais.append(texto_completo)
                count_v2 += 1
                
    print(f"✅ Claude V2: {count_v2} conversas (contas, memória de nome, fatos e código) incorporadas.")

# 3. Salvar dataset consolidado final
with open(FINAL_FILE, 'w', encoding='utf-8') as f:
    for ex in amostras_finais:
        f.write(json.dumps({"treino_formatado": ex}, ensure_ascii=False) + "\n")

print("=" * 70)
print(f"🎉 DATASET MESTRE FINAL PRONTO COM {len(amostras_finais)} AMOSTRAS!")
print(f"📁 Salvo em: {FINAL_FILE}")
print("=" * 70)
