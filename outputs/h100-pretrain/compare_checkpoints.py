"""Reproducible local generation comparison; generated code is never executed."""
import hashlib,json,time,ast,sys
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Brain,Config
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'cerebro-v1'/'trained'/'brain'
OUT=ROOT/'evaluation'; OUT.mkdir(exist_ok=True)
CASES=[
 ('geography','O Brasil é um país localizado na'),
 ('explanation','A inteligência artificial pode ajudar as pessoas a'),
 ('greeting','Pergunta: Olá! Como você está?\nResposta:'),
 ('arithmetic','Pergunta: Quanto é 2 + 2?\nResposta:'),
 ('python','def soma(a, b):\n    '),
 ('instruction','Pergunta: Responda apenas com a palavra azul.\nResposta:'),
 ('self_knowledge','Pergunta: Você consegue treinar outra IA sozinho?\nResposta:'),
 ('tool_plan','Pergunta: Antes de iniciar um treinamento pago, quais verificações você deve fazer?\nResposta:'),
 ('json','Pergunta: Escreva um JSON válido com a chave "acao" e o valor "consultar_status".\nResposta:'),
 ('english','Question: What is the capital of France?\nAnswer:')]
torch.set_num_threads(4)
reports={}
for name,folder in [('before',OLD),('after',ROOT/'result')]:
    if name=='before' and '--after-only' in sys.argv:
        reports[name]=json.loads((OUT/'before.json').read_text(encoding='utf-8')); continue
    path=folder/'best-model.pt'; tokenizer_path=folder/'tokenizer.json'
    if not tokenizer_path.exists(): tokenizer_path=ROOT/'data/corpus/tokenizer.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(tokenizer_path)!='53b3785adbcff2c283e868e1cd096cce7e1250fa7521db3aeb4ddd77a972fc7e':
        raise RuntimeError('Unexpected tokenizer')
    torch.manual_seed(42)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if name=='after':
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if saved.get('tokens')!=report['tokens_processed']: raise RuntimeError('Unexpected selected checkpoint token count')
    model=Brain(Config(**saved['config'])); model.load_state_dict(saved['model']); model.eval()
    if not all(torch.isfinite(p).all().item() for p in model.parameters()): raise RuntimeError('Nonfinite weights')
    tok=Tokenizer.from_file(str(tokenizer_path))
    result={'checkpoint_sha256':digest(path),'tokens':saved.get('tokens'),'parameters':model.parameter_count(),
            'seed':42,'new_tokens':48,'temperature':.7,'top_k':40,'results':[]}
    for case,prompt in CASES:
        tick=time.perf_counter(); ids=torch.tensor([tok.encode(prompt).ids])
        generated=model.generate(ids,new_tokens=48,temperature=.7,top_k=40)
        continuation=tok.decode(generated[0,ids.shape[1]:].tolist())
        item={'case':case,'prompt':prompt,'continuation':continuation,'seconds':round(time.perf_counter()-tick,2)}
        if case=='json':
            try: item['exact_json_correct']=json.loads(continuation.strip())=={'acao':'consultar_status'}
            except (ValueError,TypeError): item['exact_json_correct']=False
        if case=='instruction': item['exact_instruction_correct']=continuation.strip().lower()=='azul'
        if case=='python':
            try: ast.parse(prompt+continuation); item['syntax_valid']=True
            except SyntaxError: item['syntax_valid']=False
        result['results'].append(item)
        (OUT/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'model':name,**item},ensure_ascii=True),flush=True)
    reports[name]=result
    del model,saved
(OUT/'comparison.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
