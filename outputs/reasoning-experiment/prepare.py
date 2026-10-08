"""Deterministic verified comparisons; no API teacher needed."""
import hashlib,json,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def build():
    rng=random.Random(29103); used=set(); splits={}
    for split,n in [('train',100),('dev',5),('test',10)]:
        rows=[]
        for family in ['loss','budget','promotion','gpu','addition']:
            for i in range(n):
                while True:
                    a=rng.randrange(12,90); d=rng.randrange(1,10); tag=(family,a,d)
                    if tag not in used: used.add(tag); break
                for v in [0,1]:
                    if family=='loss':
                        b=a+(-d if v==0 else d)
                        question=f'Loss anterior {a/10:.1f}; loss atual {b/10:.1f}. Mesmo conjunto. Houve redução?'
                        alternate=f'No conjunto de validação fixo, o erro era {a/10:.1f} e agora é {b/10:.1f}. O erro diminuiu?'
                        answer='sim' if b<a else 'não'; step=f'{b/10:.1f} {"<" if b<a else ">"} {a/10:.1f}'
                    elif family=='budget':
                        b=a+(-d if v==0 else d)
                        question=f'Limite {a} minutos; usado {b} minutos. Encerre ao atingir ou ultrapassar o limite. Escolha continuar ou encerrar.'
                        alternate=f'O orçamento permite {a} minutos e já foram consumidos {b}. Se o consumo atingir ou superar o orçamento, pare. Escolha continuar ou encerrar.'
                        answer='continuar' if b<a else 'encerrar'; step=f'{b} {"<" if b<a else ">"} {a}'
                    elif family=='promotion':
                        b=a+(-d if v==0 else d)
                        question=f'Acertos anteriores {a}; novos {b}; regressões 0. Promova só com mais acertos e zero regressões. Escolha promover ou manter.'
                        alternate=f'A versão antiga acertou {a} questões e a nova {b}, sem regressões. Só é permitido promover com aumento de acertos. Escolha promover ou manter.'
                        answer='promover' if b>a else 'manter'; step=f'{b} {"<" if b<a else ">"} {a}; regressões = 0'
                    elif family=='gpu':
                        state='disponível' if v==0 else 'indisponível'
                        question=f'Experimento {a}-{d}: GPU {state}. Só treine com GPU disponível. Escolha TREINAR ou AGUARDAR.'
                        alternate=f'Execução {a}-{d}: a placa de vídeo está {state}. É proibido treinar quando ela estiver indisponível. Selecione TREINAR ou AGUARDAR.'
                        answer='TREINAR' if v==0 else 'AGUARDAR'; step=f'GPU = {state}'
                    else:
                        b=d+v*10
                        question=f'Calcule {a} + {b}.'; alternate=f'Some os números {a} e {b}.'
                        answer=str(a+b); step=f'{a} + {b} = {a+b}'
                    if split=='test' and i>=5: question=alternate
                    rows.append({'id':f'{split}-{family}-{i}-{v}','pair_id':f'{split}-{family}-{i}','family':family,'question':question,'answer':answer,'step':step,'phrasing':'new' if split=='test' and i>=5 else 'familiar'})
        rng.shuffle(rows); splits[split]=rows
    assert len(splits['train'])==1000 and len(splits['dev'])==50 and len(splits['test'])==100
    prompts=[{r['question'] for r in rows} for rows in splits.values()]
    assert all(not prompts[i]&prompts[j] for i in range(3) for j in range(i+1,3))
    return splits
def main():
    splits=build()
    for split,rows in splits.items():
        (ROOT/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    # Replay is identical for both arms and does not contain the new tests.
    p=ROOT.parent/'local-run-20261007-212517-0aec1e/train.jsonl'
    replay=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
    (ROOT/'replay.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in replay),encoding='utf-8')
    payload={**splits,'replay':replay}
    (ROOT/'payload.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    kernel=ROOT.parent.parent/'work/kaggle-reasoning'
    kernel.mkdir(exist_ok=True)
    trainer=(ROOT/'trainer.py').read_text(encoding='utf-8')
    (kernel/'run.py').write_text('DATA = '+repr(payload)+'\n'+trainer,encoding='utf-8')
    print(json.dumps({'train':1000,'dev':50,'test':100,'replay':len(replay),'api_calls':0,'payload_sha256':hashlib.sha256((ROOT/'payload.json').read_bytes()).hexdigest()}))
if __name__=='__main__': main()
