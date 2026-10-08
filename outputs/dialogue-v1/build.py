"""Build reviewed dialog pilot, bounded Kaggle code, and transparent provenance."""
import collections,hashlib,json,random,zlib,base64,shutil
from pathlib import Path
from tokenizers import Tokenizer
from pipeline import ROOT,WORKSPACE,read,dump,key,sha

def main():
    gate=json.loads((ROOT/'sample-review.json').read_text(encoding='utf-8'))
    if not gate.get('passed'): raise RuntimeError('Reviewed sample must pass first')
    dialogs=read(ROOT/'dialogs.jsonl'); excluded=set(gate.get('exclude_ids',[]))
    contextual=[r for r in dialogs if r['split']=='train' and r['category'].startswith('grounded_') and r['id'] not in excluded]
    natural=[r for r in dialogs if r['split'] in ['train','reserve'] and not r['category'].startswith('grounded_') and r['id'] not in excluded]
    train_dialogs=natural[:1000-len(contextual)]+contextual
    dev_dialogs=[r for r in dialogs if r['split']=='dev' and r['id'] not in excluded]
    if len(train_dialogs)!=1000 or len(dev_dialogs)<90: raise RuntimeError(f'Insufficient valid dialogs: {len(train_dialogs)}, dev {len(dev_dialogs)}')
    tok=Tokenizer.from_file(str(WORKSPACE/'outputs/h100-pretrain/result/tokenizer.json'))
    def examples(ds):
        result=[]
        for r in ds:
            p='Pergunta: '+r['question']+'\nResposta:'
            result.append({'id':r['id']+'-1','category':r['category'],'prompt':p,'response':r['answer'],'source':r['source'],'revision':r['revision'],'source_row':r['source_row'],'teacher':r['teacher'],'codex_revision':r.get('codex_revision',False),'sample_reviewed':int(r['id'].split('-')[-1])<100,'quality':'Teacher regenerated; automatic filters; first 100 reviewed separately'})
            if r['with_followup']:
                p+=' '+r['answer']+'\nPergunta: '+r['followup']+'\nResposta:'
                result.append({**result[-1],'id':r['id']+'-2','prompt':p,'response':r['followup_answer']})
        return result
    train=examples(train_dialogs); dev=examples(dev_dialogs)
    # A small replay subset prevents short-label templates dominating full replies again.
    replay=read(WORKSPACE/'outputs/local-run-20261007-212517-0aec1e/train.jsonl')
    rng=random.Random(810092); rng.shuffle(replay); replay=replay[:240]
    test=read(ROOT/'test.jsonl'); retention=read(WORKSPACE/'outputs/teacher-v2/validation.jsonl')
    held={key(r['prompt']) for r in dev+test+retention}
    for folder_name in ['teacher-v2','teacher-v3','paired-evaluation','language-pilot','kaggle-sft-review']:
        for file_name in ['audit.jsonl','validation.jsonl','test.jsonl','cases.jsonl']:
            file=WORKSPACE/'outputs'/folder_name/file_name
            if file.exists(): held.update(key(r['prompt']) for r in read(file))
    replay=[r for r in replay if key(r['prompt']) not in held]
    train=[r for r in train if key(r['prompt']) not in held]
    train=list({key(r['prompt']):r for r in train}.values())
    for r in train+dev+replay: assert len(tok.encode(r['prompt']).ids+tok.encode(r['response']+'<|eos|>').ids)<=512
    payload={'train':train,'dev':dev,'test':test,'replay':replay,'retention':retention}
    for name,rows in payload.items(): dump(ROOT/(name+'.jsonl'),rows)
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    calls=read(ROOT/'calls.jsonl')
    manifest.update(dialogs_train=1000,dialogs_dev=len(dev_dialogs),counts={k:len(v) for k,v in payload.items()},train_response_tokens=sum(len(tok.encode(r['response']+'<|eos|>').ids) for r in train),categories=dict(collections.Counter(r['category'] for r in train_dialogs)),api_calls=len(calls),api_total_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in calls),sample_gate=gate,sha256={k:sha(ROOT/(k+'.jsonl')) for k in payload},teacher_review_limits='Same Groq teacher reviewed sample; Codex separately read first 100. Remaining dialogs validated structurally, not independently certified.',training_recipe={'epochs':8,'seconds':900,'batch':8,'lr':3e-5,'stop_after_validation_without_improvement':2},automatic_promotion=False)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'payload.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    trainer=(WORKSPACE/'outputs/language-pilot/trainer.py').read_text(encoding='utf-8')
    trainer=trainer.replace("Path('/kaggle/working/language-pilot')","Path('/kaggle/working/dialogue-pilot')")
    trainer=trainer.replace('<=384','<=512').replace('61820','810093')
    trainer=trainer.replace('range(0,len(dev),16)','range(0,len(dev),8)').replace('dev[start:start+16]','dev[start:start+8]')
    trainer=trainer.replace('80 if','120 if')
    trainer=trainer.replace('def evaluate(rows,label):','def evaluate(rows,label,temperature=0):')
    trainer=trainer.replace("with torch.autocast('cuda',dtype=torch.float16): value=net(x)[:,-1].argmax(-1,keepdim=True)","with torch.autocast('cuda',dtype=torch.float16): logits=net(x)[:,-1].float()\n                if temperature:\n                    logits=logits/temperature; cutoff=logits.topk(40).values[:,-1:]; logits=logits.masked_fill(logits<cutoff,float('-inf'))\n                    value=torch.multinomial(logits.softmax(-1),1)\n                else: value=logits.argmax(-1,keepdim=True)")
    trainer=trainer.replace("report['before']=evaluate(DATA['test'],'before'); report['retention_before']", "report['before']=evaluate(DATA['test'],'before'); torch.manual_seed(43); report['before_sampling']=evaluate([r for r in DATA['test'] if r['scorer']=='manual'][:8],'before-sampling',.7); report['retention_before']")
    trainer=trainer.replace("report['after']=evaluate(DATA['test'],'after'); report['retention_after']", "report['after']=evaluate(DATA['test'],'after'); torch.manual_seed(43); report['after_sampling']=evaluate([r for r in DATA['test'] if r['scorer']=='manual'][:8],'after-sampling',.7); report['retention_after']")
    trainer=trainer.replace("'api_teacher_calls':0","'api_teacher_calls':MANIFEST['api_calls']")
    trainer=trainer.replace("'epochs':4,'seconds':600,'batch':16,'learning_rate':2e-5","'epochs':8,'seconds':900,'batch':8,'learning_rate':3e-5")
    trainer=trainer.replace('lr=2e-5','lr=3e-5').replace('range(4)','range(8)').replace('range(0,len(order),16)','range(0,len(order),8)').replace('cursor:cursor+16','cursor:cursor+8').replace('>=600','>=900')
    # Open-response gate is primary and must be applied AFTER blinded review.
    trainer=trainer.replace("report['after']['correct']>=report['before']['correct']+8 and not regressions","report['after']['correct']>=report['before']['correct'] and not regressions")
    (ROOT/'trainer.py').write_text(trainer,encoding='utf-8')
    folder=WORKSPACE/'work/kaggle-dialogue-pilot'; folder.mkdir(exist_ok=True)
    packed=base64.b64encode(zlib.compress(json.dumps({'DATA':payload,'MANIFEST':manifest},ensure_ascii=False).encode(),9)).decode()
    prefix="import base64,zlib,json\n_blob=json.loads(zlib.decompress(base64.b64decode("+repr(packed)+")))\nDATA=_blob['DATA']; MANIFEST=_blob['MANIFEST']\n"
    (folder/'run.py').write_text(prefix+trainer,encoding='utf-8')
    meta=json.loads((WORKSPACE/'work/kaggle-language-pilot/kernel-metadata.json').read_text(encoding='utf-8'))
    meta.update(id='andersonluan10ad/cerebro-102m-dialogue-pilot-oct08',title='Cerebro 102M Dialogue Pilot Oct08')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    assert (folder/'run.py').stat().st_size<1000000
    (ROOT/'README.md').write_text('''# Conversas para o modelo próprio de 102M

1.000 diálogos com perguntas diversas, parte com uma continuação, e validação separada. Perguntas adaptadas do Instruct-Aira v2 (Nicholas Kluge Corrêa, Apache-2.0); revisão fixada no manifest. Todas as respostas foram reescritas pelo professor Groq GPT-OSS-120B. As respostas antigas do repositório não foram usadas como alvo. Exemplos de criação fictícia não significam execução externa ou experiências pessoais do assistente.

A primeira seleção foi rejeitada por qualidade. A segunda exclui perguntas especializadas sem contexto, recomendações comerciais, respostas que contêm reject e pedidos incompatíveis com tamanho curto. A primeira amostra é lida pelo Codex e revisada também pelo professor. Esta é revisão por IAs, não revisão humana independente. Os demais diálogos têm filtros de tamanho, estrutura e duplicação; não se presume perfeição factual.

O histórico usa o formato do chat existente: Pergunta/Resposta, com respostas anteriores no contexto e loss apenas na resposta atual. O replay limitado conserva tarefas sem dominar as respostas completas. O modelo continua sendo nosso próprio 102.257.280 parâmetros.

Testes congelados antes do treino: 32 respostas abertas, 24 objetivas (incluindo controle de ordem da negação), 12 diagnósticos antigos de retenção. A seleção do checkpoint usa somente validação. Não promover pelo valor da loss: exigir melhora de ao menos oito respostas úteis, no mínimo 16/32 abertas e nenhuma regressão nos 12 antigos, sem queda objetiva importante. Revisão cega das respostas e verificação pelo Codex. Os temas podem coincidir com o treino; não é prova de autonomia nem um benchmark geral independente.

Kaggle: até oito épocas, 900 segundos no laço, batch global 8, LR 3e-5, CUDA/duas T4, sem fallback para CPU; parada depois de duas validações sem melhora; checkpoint candidato separado. Limite da sessão 1.800 segundos. Não ativa automaticamente candidato ou treino maior.

Comandos de preparação reutilizam os caches e têm limites por chamada e total; não chamam APIs novamente para lotes já salvos. Não contém credenciais. Todos os professores usam segredos DPAPI locais temporariamente. A geração não recebe os testes. Nenhum script de professor executa comandos escritos pelo modelo.
''',encoding='utf-8')
    print(json.dumps({'event':'ready','dialogs':1000,'counts':manifest['counts'],'response_tokens':manifest['train_response_tokens'],'api_calls':manifest['api_calls'],'api_total_tokens':manifest['api_total_tokens'],'kernel_bytes':(folder/'run.py').stat().st_size},ensure_ascii=False))
if __name__=='__main__': main()
