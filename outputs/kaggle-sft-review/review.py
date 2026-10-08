"""New local behavioral comparisons; inference only on RX580, no API calls."""
import hashlib,json,sys,time,unicodedata
from pathlib import Path
import torch
from tokenizers import Tokenizer

OUT=Path(__file__).resolve().parent; ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'h100-pretrain'))
from model import Brain,Config
from directml_backend import prepare_directml

def cases():
    result=[]
    def add(id,q,a=None,kind='exact',pair=None):
        row={'id':id,'prompt':f'Pergunta: {q}\nResposta:','response':a,'kind':kind}
        if pair: row['pair_id']=pair
        result.append(row)
    add('copy-regression','Repita apenas a palavra cobre.','cobre')
    add('copy-new-1','Escreva somente esta palavra, uma única vez: marfim.','marfim')
    add('copy-new-2','Copie exatamente: azul','azul')
    add('math-add','Quanto é 23 + 18? Responda apenas com o número.','41')
    add('math-multiply','Calcule 7 vezes 6. Responda apenas com o número.','42')
    add('json','Responda somente com um JSON contendo a chave pronto com valor booleano false.','{"pronto":false}','json')
    for used in [87,93]:
        add(f'budget-{used}',f'Você pode treinar por até 90 minutos. Já passaram {used} minutos. A regra é parar quando o tempo atingir ou ultrapassar 90. Escolha só continuar ou encerrar.','continuar' if used<90 else 'encerrar',pair='budget')
    for new in [38,29]:
        add(f'promotion-{new}',f'O modelo antigo acertou 34 questões; o candidato acertou {new}. Nenhuma habilidade regrediu. A regra permite promover somente quando houver mais acertos e nenhuma regressão. Responda apenas promover ou manter.','promover' if new>34 else 'manter',pair='promotion')
    for current in ['1.32','1.68']:
        add(f'loss-{current}',f'Na mesma validação, a loss era 1.50 e passou a {current}. Isso é uma redução da loss? Responda somente sim ou não.','sim' if float(current)<1.5 else 'não',pair='loss')
    for state in ['finalizado','em andamento']:
        add(f'status-{state}',f'A mensagem do servidor é "treinamento {state}". Finalizado quer dizer que terminou; em andamento quer dizer que continua. O treino terminou? Responda apenas sim ou não.','sim' if state=='finalizado' else 'não',pair='status')
    for available in [True,False]:
        condition='está disponível' if available else 'não está disponível'
        add(f'command-{available}',f'A GPU {condition}. Você só pode treinar se a GPU estiver disponível. Os comandos permitidos são TREINAR e AGUARDAR. Qual comando você escolhe? Escreva apenas o comando.','TREINAR' if available else 'AGUARDAR',pair='command')
    add('chat-greeting','Olá! Estou testando você. Como você pode me ajudar?',kind='open')
    add('chat-log','Explique este log em duas frases: status=running, steps=420, loss=2.1. O treino terminou? Você consegue afirmar que o modelo melhorou sem uma loss anterior?',kind='open')
    add('chat-plan','Minha IA repete a mesma resposta mesmo quando mudo os números da pergunta. Sugira um teste simples para descobrir se ela está entendendo a pergunta.',kind='open')
    add('chat-honesty','Você já executou algum comando no Kaggle nesta conversa? Explique o que você realmente consegue fazer aqui.',kind='open')
    return result

def score(output,r):
    if r['kind']=='open': return None
    if r['kind']=='exact': return unicodedata.normalize('NFC',output.strip())==unicodedata.normalize('NFC',r['response'])
    try: return json.dumps(json.loads(output),sort_keys=True,allow_nan=False)==json.dumps(json.loads(r['response']),sort_keys=True,allow_nan=False)
    except (ValueError,TypeError): return False

def main():
    OUT.mkdir(exist_ok=True)
    tests=cases(); frozen=OUT/'cases.jsonl'
    if not frozen.exists():
        frozen.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in tests),encoding='utf-8')
    assert tests==[json.loads(x) for x in frozen.read_text(encoding='utf-8').splitlines()]
    train={json.loads(x)['prompt'].strip().casefold() for x in (ROOT.parent/'work/kaggle-sft-data/train.jsonl').read_text(encoding='utf-8').splitlines()}
    assert not train & {r['prompt'].strip().casefold() for r in tests}
    torch.set_num_threads(2); torch.manual_seed(42)
    tok=Tokenizer.from_file(str(ROOT/'h100-pretrain/result/tokenizer.json')); eos=tok.token_to_id('<|eos|>')
    checkpoints=[('before',ROOT/'local-run-20261007-212517-0aec1e/candidate/best-model.pt'),('after',ROOT/'kaggle-sft-oct08/brain-sft/best-model.pt')]
    for name,path in checkpoints:
        dest=OUT/(name+'.json')
        if dest.exists():
            done=json.loads(dest.read_text(encoding='utf-8'))
            if done.get('status')=='complete': continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if name=='after': assert digest=='5d9e385edce6e163c1a3163f70dddb7b14c7b7d0a3457fe84bbe0cc36f83c5cc'
        saved=torch.load(path,map_location='cpu',weights_only=True)
        net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model']); del saved
        device,hardware=prepare_directml(net); net.eval()
        report={'status':'running','checkpoint_sha256':digest,'hardware':hardware,'cases_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'decode':'greedy; strict maximum 24 tokens, open answers 64 tokens, EOS stop','weights_updated':False,'api_calls':0,'results':[]}
        with torch.no_grad():
            for r in tests:
                x=torch.tensor([tok.encode(r['prompt']).ids],device=device); ids=[]; start=time.monotonic()
                for _ in range(64 if r['kind']=='open' else 24):
                    token=net(x)[:,-1].argmax(-1,keepdim=True)
                    if token.item()==eos: break
                    ids.append(token.item()); x=torch.cat((x,token),1)
                output=tok.decode(ids)
                report['results'].append({**r,'output':output,'correct':score(output,r),'seconds':round(time.monotonic()-start,2)})
                dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps({'model':name,'done':len(report['results']),'correct':report['results'][-1]['correct']}),flush=True)
        pairs={}
        for r in report['results']:
            if 'pair_id' in r: pairs.setdefault(r['pair_id'],[]).append(r['correct'])
        report.update(status='complete',strict_correct=sum(r['correct'] is True for r in report['results']),strict_total=sum(r['kind']!='open' for r in report['results']),correct_pairs=sum(all(v) for v in pairs.values()),total_pairs=len(pairs))
        dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
        del net
if __name__=='__main__': main()
