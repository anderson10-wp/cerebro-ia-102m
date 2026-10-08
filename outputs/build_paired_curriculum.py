"""128 original training cases, 32 separate cases; no API or execution."""
import hashlib,json,random
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
def main():
    rng=random.Random(8181); splits={}
    for split,count,start in [('train',16,0),('validation',4,100)]:
        rows=[]
        for i in range(count):
            n=start+i
            # Both members have the same wording and differ in just one value.
            a=200+n*7; delta=3+i%5
            loss=[(f'Loss anterior: {a/100:.2f}. Loss atual: {(a+d)/100:.2f}. Mesmo conjunto. A loss caiu? Só sim ou não.', 'sim' if d<0 else 'não') for d in [-delta,delta]]
            if split=='validation':
                loss=[(q.replace('Mesmo conjunto. A loss caiu? Só sim ou não.','Compare no mesmo conjunto de validação: houve redução? Responda sim ou não.'),a) for q,a in loss]
            old=20+n; new=old+2
            promotion=[(f'Anterior: {old} acertos. Novo: {new} acertos. Regressões: {r}. Regra: promover só com mais acertos e zero regressões. Responda promover ou manter.','promover' if r==0 else 'manter') for r in [0,1]]
            limit=30+n
            budget=[(f'Limite: {limit} minutos. Tempo usado: {used} minutos. Regra: encerrar ao atingir o limite. Responda continuar ou encerrar.','continuar' if used<limit else 'encerrar') for used in [limit-1,limit]]
            status=[(f'Log: {{"status":"{s}","steps":{200+n}}}. completed significa concluído; running significa em execução. O treino terminou? Só sim ou não.','sim' if s=='completed' else 'não') for s in ['completed','running']]
            if split=='validation':
                promotion=[(q.replace('Anterior:','Versão anterior:').replace('Novo:','Candidato:'),a) for q,a in promotion]
                budget=[(q.replace('Limite:','Orçamento de tempo:').replace('Tempo usado:','Tempo consumido:'),a) for q,a in budget]
                status=[(q.replace('Log:','Registro recebido:').replace('O treino terminou?','A execução foi concluída?'),a) for q,a in status]
            for family,pair in [('loss',loss),('promotion',promotion),('budget',budget),('status',status)]:
                rng.shuffle(pair)
                for j,(q,a) in enumerate(pair):
                    rows.append(dict(id=f'paired-{split}-{family}-{n}-{j}',pair_id=f'{split}-{family}-{n}',family=family,split=split,category='training_operations_paired',prompt=f'Pergunta: {q}\nResposta:',response=a,scorer='exact',checked=True,validation_level='deterministic explicit condition; paired opposite labels',source='Original local curriculum; proposal 20261008-083638-15dffd'))
        rng.shuffle(rows); splits[split]=rows
    tok=Tokenizer.from_file(str(ROOT/'h100-pretrain/result/tokenizer.json'))
    trainkeys={r['prompt'].strip().casefold() for r in splits['train']}
    held={r['prompt'].strip().casefold() for r in splits['validation']}
    for file in [ROOT/'teacher-v2/validation.jsonl',ROOT/'teacher-v2/audit.jsonl',ROOT/'hf-curriculum/technical-validation.jsonl']:
        held.update(json.loads(line)['prompt'].strip().casefold() for line in file.read_text(encoding='utf-8').splitlines())
    assert not trainkeys&held and len(trainkeys)==128
    for rows in splits.values():
        pairs={}
        for r in rows:
            assert len(tok.encode(r['prompt']+r['response']+'<|eos|>').ids)<180
            pairs.setdefault(r['pair_id'],[]).append(r)
        assert all(len(p)==2 and len({r['response'] for r in p})==2 for p in pairs.values())
    trainfolder=ROOT/'curriculum-teacher/batch-002'; testfolder=ROOT/'paired-evaluation'
    trainfolder.mkdir(parents=True,exist_ok=True); testfolder.mkdir(exist_ok=True)
    for folder,split in [(trainfolder,'train'),(testfolder,'validation')]:
        path=folder/f'{split}.jsonl'
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in splits[split])
        if path.exists() and path.read_text(encoding='utf-8')!=content: raise ValueError('Frozen dataset differs; create a new version')
        path.write_text(content,encoding='utf-8')
    report={'status':'completed','accepted':128,'training_pairs':64,'reserved_cases':32,'reserved_pairs':16,'api_calls':0,'training_started':False,'train_sha256':hashlib.sha256((trainfolder/'train.jsonl').read_bytes()).hexdigest(),'validation_sha256':hashlib.sha256((testfolder/'validation.jsonl').read_bytes()).hexdigest(),'constant_baseline':{'individual_accuracy':0.5,'pair_accuracy':0.0},'limitations':'Narrow synthetic tasks; related templates, not broad independent benchmark. No capability claim until evaluated.'}
    (trainfolder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (testfolder/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
if __name__=='__main__': main()
