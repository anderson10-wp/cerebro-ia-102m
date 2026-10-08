import hashlib,json
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent; OUT=ROOT.parent
COURSE='https://huggingface.co/learn/llm-course/chapter7/6'
def save(path,rows): path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def main():
    records={r['row_idx']:r['row'] for r in json.loads((ROOT/'sample.json').read_text(encoding='utf-8'))}
    # Individually reviewed and rewritten short responses, not automatically trusted labels.
    answers={2:'Um polígono é uma figura plana fechada formada por segmentos de reta. Triângulos e quadrados são exemplos.',9:'Um verbo é uma palavra que expressa ação, estado ou fenômeno. Correr, estar e chover são exemplos.',10:'Trem, avião, bicicleta e scooter.',30:'Uma pipa é um objeto leve preso a uma linha que pode voar com a ação do vento.',35:'Cadeira, mesa e micro-ondas.',46:'A maçã é o fruto da macieira. Existem variedades com diferentes cores, sabores e texturas.',63:'A feijoada é um prato típico brasileiro feito com feijão e carnes, geralmente acompanhado de arroz e farofa.',77:'Alce, cachorro, gato e humano são mamíferos. Sapo e salamandra são anfíbios. Nenhum dos animais listados é peixe.',117:'Vermelho, preto, amarelo, laranja, branco, rosa e azul.',173:'O Alasca é o maior estado dos Estados Unidos em área territorial.',186:'- Gênesis\n- Êxodo\n- Levítico\n- Números\n- Deuteronômio',438:'Primavera, verão, outono e inverno.',459:'Oito planetas: Mercúrio, Vênus, Terra, Marte, Júpiter, Saturno, Urano e Netuno.',466:'Fósseis de dinossauros são estudados por paleontólogos. Arqueólogos estudam vestígios das sociedades humanas.',476:'Um piano padrão tem 88 teclas: 52 brancas e 36 pretas.',499:'Inglaterra e Grécia ficam na Europa. Tailândia e Japão ficam na Ásia.'}
    rows=[]
    def add(identifier,category,q,a,source,validation='deterministic scenario'):
        rows.append(dict(id=identifier,split='train',category=category,prompt=f'Pergunta: {q}\nResposta:',response=a,scorer='exact',checked=True,validation_level=validation,source_url=source))
    for index,answer in answers.items():
        add(f'aira-reviewed-{index}','portuguese_conversation',records[index]['conversations'][0]['content'],answer,'https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset-v2','Codex content review and rewritten response; not independently fact-checked')
        rows[-1].update(source_row=index,license='Apache-2.0 declared by source',modified=True)
    def technical(start,count,split):
        result=[]
        for i in range(start,start+count):
            a=1+i/100; b=a-.05 if i%2 else a+.05
            questions=[(f'Experimento {i}: loss de validação anterior {a:.2f}; atual {b:.2f}, no mesmo conjunto. A loss diminuiu? Responda sim ou não.','sim' if b<a else 'não'),(f'Rodada {i}: modelo anterior acertou 6 de 12; candidato acertou {7 if i%2 else 5}. Regra: promover somente com mais acertos e sem regressões. Houve {0 if i%2 else 1} regressões. Responda promover ou manter.','promover' if i%2 else 'manter'),(f'Execução {i}: limite de {10+i} minutos, já foram usados {9+i if i%2 else 10+i}. Ao atingir o limite, encerrar. Responda continuar ou encerrar.','continuar' if i%2 else 'encerrar'),(f'Registro do treino {i}: {{"status":"{ "completed" if i%2 else "running"}","steps":{i+50}}}. O processo terminou? Responda sim ou não.','sim' if i%2 else 'não')]
            for j,(q,a) in enumerate(questions): result.append(dict(id=f'training-ops-{split}-{i}-{j}',split=split,category='training_operations',prompt=f'Pergunta: {q}\nResposta:',response=a,scorer='exact',checked=True,validation_level='deterministic comparisons and explicit policy; no commands executed',source_url='local training policy; pedagogical reference '+COURSE))
        return result
    rows+=technical(0,16,'train')
    tests=technical(71,4,'validation')
    tok=Tokenizer.from_file(str(OUT/'h100-pretrain/result/tokenizer.json'))
    held={json.loads(l)['prompt'].strip().casefold() for split in ['validation','audit'] for l in (OUT/f'teacher-v3/{split}.jsonl').read_text(encoding='utf-8').splitlines()}
    held.update(r['prompt'].strip().casefold() for r in tests)
    assert not held & {r['prompt'].strip().casefold() for r in rows}
    assert len({r['prompt'] for r in rows})==len(rows)
    assert all(len(tok.encode(r['prompt']+r['response']+'<|eos|>').ids)<=180 for r in rows+tests)
    folder=OUT/'curriculum-teacher/batch-001'; folder.mkdir(parents=True,exist_ok=True)
    save(folder/'train.jsonl',rows); save(ROOT/'technical-validation.jsonl',tests)
    report=dict(status='completed',accepted=len(rows),aira_adapted=16,technical_scenarios=64,reserved_technical_tests=16,api_calls=0,training_started=False,source_cache_sha256=hashlib.sha256((ROOT/'sample.json').read_bytes()).hexdigest(),notes='Technical examples are original local scenarios, not extracted course text. Aira sample is small and not representative. Reserved technical cases share templates; they only test narrow transfer.')
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
if __name__=='__main__': main()
