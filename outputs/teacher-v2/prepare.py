"""Small teacher-authored curriculum. No API calls, weight updates or cloud jobs."""
import hashlib,json,sys,unicodedata
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
TOKENIZER=ROOT.parent/'h100-pretrain/result/tokenizer.json'
rows=[]
def add(split,category,request,response,kind='exact'):
    rows.append(dict(id=f'{split}-{len(rows):03}',split=split,category=category,
        prompt='Pergunta: '+request+'\nResposta:',response=response,scorer=kind,
        provenance='Codex-authored and reviewed in this session',
        scope='Short response; no hidden reasoning trace; no tools executed'))

TRAIN={
'instruction':[
('Responda somente com a palavra verde.','verde'),
('Copie exatamente: janela','janela'),
('Transforme casa em letras maiúsculas. Responda só o resultado.','CASA'),
('Escreva apenas a última palavra de: gato preto dorme.','dorme'),
('Coloque em ordem alfabética: pato, abelha, leão. Separe por vírgula e espaço.','abelha, leão, pato'),
('Liste os números 1, 2 e 3, um por linha.','1\n2\n3'),
('Retire os espaços de: a b c. Escreva só o resultado.','abc'),
('Responda sim se a frase contém a palavra sol, ou não caso contrário: O sol nasceu.','sim'),
('Responda sim se a frase contém a palavra sol, ou não caso contrário: A noite chegou.','não'),
('Escreva exatamente duas palavras: bom dia.','bom dia')],
'context':[
('Texto: Lúcia vendeu três livros na terça-feira. O que Lúcia vendeu? Responda só o objeto.','três livros'),
('Texto: A reunião começa às 14h na sala B. Qual é a sala? Responda só o nome.','sala B'),
('Texto: Igor tem um cachorro chamado Pingo. Como se chama o cachorro?','Pingo'),
('Texto: O pacote pesa 5 kg e custa 30 reais. Quanto ele pesa?','5 kg'),
('Texto: Há quatro cadeiras e duas mesas. Quantas mesas há? Responda só o número.','2'),
('Texto: A senha fictícia do exercício é vento. Copie a senha fictícia.','vento'),
('Texto: Clara chegou depois de Beto. Quem chegou primeiro?','Beto'),
('Texto: O trem saiu de Recife e chegou a Natal. Qual foi o destino?','Natal'),
('Texto: A loja abre às 9h. Qual o horário de fechamento? Se o texto não disser, responda não informado.','não informado'),
('Texto: O carro de Ana é amarelo. Qual é a cor da bicicleta dela? Se faltar essa informação, responda não informado.','não informado')],
'limits':[
('Como está o treinamento agora? Não há acesso a ferramentas nem relatório nesta conversa.','Não tenho o estado atual do treinamento. Preciso de um relatório ou de uma ferramenta de consulta.'),
('Você consegue iniciar uma GPU nesta conversa? Nenhuma ferramenta foi conectada.','Não. Nesta conversa, posso gerar texto, mas não iniciar uma GPU.'),
('Você aprendeu permanentemente o que eu acabei de escrever? Nenhum treinamento foi executado.','Não. Uma mensagem pode entrar no contexto, mas não altera meus pesos automaticamente.'),
('Melhore isso.','O que você quer melhorar? Envie o texto, código ou uma descrição do problema.'),
('Quanto vai custar?','Qual serviço você quer usar e por quanto tempo?'),
('Compare os dois modelos. Não foram enviados nomes nem resultados.','Quais são os dois modelos e quais resultados devemos comparar?'),
('O teste não foi executado. Posso dizer que o modelo passou?','Não. É preciso executar o teste e verificar o resultado.'),
('A resposta parece bem escrita. Isso garante que esteja correta?','Não. Fluência não garante que a informação esteja correta.')],
'dialogue':[
('Olá!','Olá! Como posso ajudar?'),
('Obrigado pela ajuda.','De nada!'),
('Não entendi a resposta.','Qual parte ficou confusa? Posso explicar de outra forma.'),
('Você respondeu errado: 2 + 2 é 4.','Você tem razão: 2 + 2 = 4.'),
('Explique em uma frase o que é um tokenizer.','Um tokenizer transforma texto em unidades chamadas tokens, que o modelo processa.'),
('Explique em uma frase para que serve um teste separado do treino.','Ele verifica se o modelo consegue responder a exemplos que não foram usados para ajustar seus pesos.')],
'code':[
('Escreva apenas uma função Python soma(a, b) que devolva a soma dos argumentos.','def soma(a, b):\n    return a + b'),
('Escreva apenas uma função Python dobro(x) que devolva o dobro de x.','def dobro(x):\n    return 2 * x'),
('Complete somente a linha: return a __ b. O objetivo é subtrair b de a.','return a - b'),
('Qual o resultado de len([4, 8, 9]) em Python? Responda só o número.','3'),
('Corrija apenas a linha Python: print("oi"','print("oi")'),
('Em Python, escreva apenas uma expressão que teste se x é igual a 7.','x == 7')]
}
for category,examples in TRAIN.items():
    for request,response in examples: add('train',category,request,response)
for a,b,op in [(3,5,'+'),(12,7,'+'),(8,6,'+'),(21,9,'-'),(10,4,'-'),(30,18,'-'),(3,4,'*'),(6,7,'*'),(5,8,'*'),(18,3,'/'),(24,6,'/'),(35,5,'/')]:
    result={'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a//b}[op]()
    add('train','arithmetic',f'Calcule {a} {op} {b}. Responda apenas o número.',str(result))
for field,value in [('cor','roxo'),('animal','gato'),('cidade','Recife'),('estado','pendente'),('acao','consultar_status'),('resultado','ok')]:
    add('train','json',f'Responda somente com um objeto JSON com a chave "{field}" e o valor "{value}".',json.dumps({field:value},ensure_ascii=False),'json')
for request,response in [
('Simulação. Ferramentas disponíveis: consultar_status. Para saber o estado real do treino, produza apenas o JSON com ferramenta e argumentos. Não execute nada.',{'ferramenta':'consultar_status','argumentos':{}}),
('Simulação. Ferramentas disponíveis: calculadora. Calcule 17 + 6 produzindo só o JSON com ferramenta e argumentos; o argumento expressao deve copiar "17 + 6".',{'ferramenta':'calculadora','argumentos':{'expressao':'17 + 6'}}),
('Simulação. Ferramentas disponíveis: professor. Encaminhe a pergunta "O que é validação?" ao professor. Responda só JSON com ferramenta e argumentos; use a chave pergunta.',{'ferramenta':'professor','argumentos':{'pergunta':'O que é validação?'}}),
('Simulação. Resultado fornecido pela ferramenta: {"estado":"concluido"}. Qual o estado? Responda só concluido ou desconhecido.',None)]:
    if response is None: add('train','tool_result',request,'concluido')
    else: add('train','tool_simulation',request,json.dumps(response,ensure_ascii=False),'json')
add('train','tool_result','Simulação. Resultado da calculadora: {"resultado":23}. Copie somente o número retornado.','23')
add('train','tool_result','Simulação. A consulta retornou erro e nenhum estado. Qual o estado do treino? Responda só desconhecido.','desconhecido')

# Hand-authored held-out inputs. No expanded template siblings are shuffled into these splits.
DEV=[
('instruction','Sua saída deve conter só: cobre','cobre','exact'),
('instruction','Qual é a primeira palavra de "rio largo corre"? Escreva apenas essa palavra.','rio','exact'),
('context','Bilhete: Rui deixou a chave na gaveta. Onde está a chave? Responda apenas o local.','na gaveta','exact'),
('context','Ficha: nome Lia; idade não registrada. Qual é a idade? Use "não informado" quando faltar o dado.','não informado','exact'),
('arithmetic','Dê somente o valor de 16 + 27.','43','exact'),
('arithmetic','Sem explicação, quanto vale 9 * 8?','72','exact'),
('json','Devolva o objeto JSON {"modo":"teste"} sem texto ao redor.','{"modo":"teste"}','json'),
('json','Use JSON para representar uma única chave ativo com valor booleano falso. Nenhum outro campo.','{"ativo":false}','json'),
('tool_simulation','Exercício simulado: só existe consultar_status. Gere o objeto JSON {"ferramenta":"consultar_status","argumentos":{}} para consultar o treino. Não execute a consulta.','{"ferramenta":"consultar_status","argumentos":{}}','json'),
('tool_result','Retorno simulado: {"resultado":91}. Diga apenas o resultado numérico.','91','exact'),
('code','Em Python, qual o valor de len([])? Retorne somente o número.','0','exact'),
('instruction','Escreva apenas a segunda palavra: manhã calma.','calma','exact')]
AUDIT=[
('instruction','Imprima unicamente a palavra tijolo.','tijolo','exact'),
('instruction','Leia "vento forte passou" e devolva só a palavra do meio.','forte','exact'),
('context','Registro: o ingresso pertence a Nara; o assento é C8. Qual é o assento? Copie apenas o código.','C8','exact'),
('context','No texto "O barco saiu hoje", o destino foi informado? Responda somente sim ou não.','não','exact'),
('arithmetic','Qual número completa 54 - 29 = ? Não escreva mais nada.','25','exact'),
('arithmetic','Responda numericamente: 72 dividido por 8.','9','exact'),
('json','A saída inteira deve ser um objeto JSON contendo somente quantidade igual ao número 4.','{"quantidade":4}','json'),
('json','Crie um objeto JSON com uma única chave itens cujo valor seja uma lista vazia.','{"itens":[]}','json'),
('tool_result','Uma ferramenta fictícia devolveu {"estado":"pausado"}. Escreva só o valor de estado.','pausado','exact'),
('tool_simulation','Simulação sem ferramentas disponíveis. Há evidência para afirmar que um treino foi iniciado? Responda somente sim ou não.','não','exact'),
('code','Qual é o valor da expressão Python 5 == 5? Escreva somente o booleano Python.','True','exact'),
('instruction','Da sequência 6, 2, 9, devolva somente o menor número.','2','exact')]
for split,examples in [('validation',DEV),('audit',AUDIT)]:
    for category,request,response,kind in examples: add(split,category,request,response,kind)

def main():
    tok=Tokenizer.from_file(str(TOKENIZER)); normalized=set(); report={}
    for row in rows:
        key=unicodedata.normalize('NFKC',row['prompt']).casefold().strip()
        assert key not in normalized; normalized.add(key)
        if row['scorer']=='json': assert isinstance(json.loads(row['response']),dict)
        row['prompt_tokens']=len(tok.encode(row['prompt']).ids)
        row['response_tokens']=len(tok.encode(row['response']+'<|eos|>').ids)
        assert row['prompt_tokens']+row['response_tokens']<=1024
    for split in ('train','validation','audit'):
        subset=[r for r in rows if r['split']==split]
        path=ROOT/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset),encoding='utf-8')
        report[split]={'examples':len(subset),'response_tokens':sum(r['response_tokens'] for r in subset),
            'categories':dict(Counter(r['category'] for r in subset)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    report.update(tokenizer_sha256=hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        scope='Small seed curriculum, not a broad instruction dataset or proof of capability.',
        audit_policy='Frozen before tuning; do not train on audit or use it for checkpoint selection.',
        split_limitation='Tasks intentionally overlap; prompts differ. This is not a guarantee against semantic overlap.',
        provenance='Written by Codex in this session; not independently human-reviewed.',weights_updated=False,paid_api_calls=0)
    (ROOT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
