"""Read-only audit: checkpoint identities, tokenizer, masking, causal inference."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/'model-audit'
sys.path.insert(0,str(ROOT/'h100-pretrain'))
from model import Brain,Config
from directml_backend import prepare_directml
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def main():
    tokpaths=[ROOT/'cerebro-v1/trained/brain/tokenizer.json',ROOT/'h100-pretrain/result/tokenizer.json',ROOT/'h100-pretrain/data/corpus/tokenizer.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in tokpaths}; assert len(set(hashes.values()))==1
    tok=Tokenizer.from_file(str(tokpaths[1])); eos=tok.token_to_id('<|eos|>')
    samples=['Olá! A GPU não está disponível.','A loss caiu de 1,50 para 1,32.','Responda somente com a palavra azul.','23 + 18 = 41.','{"pronto":false}']
    tokenchecks=[]
    for text in samples:
        ids=tok.encode(text).ids; decoded=tok.decode(ids)
        tokenchecks.append({'text':text,'tokens':len(ids),'unknown_tokens':ids.count(tok.token_to_id('<|unk|>')),'roundtrip_exact':decoded==text})
    assert all(r['roundtrip_exact'] and r['unknown_tokens']==0 for r in tokenchecks)
    labelchecks=[]
    for text,answer in [('Pergunta: Repita azul.\nResposta:','azul'),('Pergunta: Some 4 e 3.\nResposta:','7')]:
        pi=tok.encode(text).ids; ai=tok.encode(answer+'<|eos|>').ids
        x=pi+ai[:-1]; y=[-100]*(len(pi)-1)+ai
        assert len(x)==len(y) and y[len(pi)-1]==ai[0] and y[-1]==eos
        labelchecks.append({'first_answer_token_supervised':True,'eos_supervised':True,'label_count':sum(t!=-100 for t in y)})
    report={'status':'running','weights_updated':False,'cloud_jobs_started':0,'api_teacher_calls':0,'tokenizer_sha256':hashes,'tokenizer_tests':tokenchecks,'label_alignment_tests':labelchecks,'base_training':read(ROOT/'h100-pretrain/result/report.json'),'corpus_counts':read(ROOT/'h100-pretrain/data/corpus/manifest.json')['counts'],'models':[],'limitations':'Spot checks and source review, not a complete corpus or numerical kernel audit. CPU loads checkpoint metadata; all model inference uses RX580.'}
    dest=OUT/'report.json'
    def save(): dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    prompts=['O Brasil é um país da América do Sul. Sua capital é','Para estudar um assunto novo, é importante','As plantas precisam de água e luz porque','Pergunta: Olá, como você pode me ajudar?\nResposta:']
    torch.set_num_threads(2); torch.manual_seed(42)
    for name,path in [('base-pretrained',ROOT/'h100-pretrain/result/best-model.pt'),('current-instruction-reference',ROOT/'local-run-20261007-212517-0aec1e/candidate/best-model.pt')]:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model'])
        identity={'name':name,'checkpoint_sha256':sha(path),'config':saved['config'],'parameters':net.parameter_count(),'all_weights_finite':all(torch.isfinite(t).all().item() for t in saved['model'].values()),'generations':[]}
        assert identity['parameters']==102257280 and identity['all_weights_finite']
        del saved; device,hardware=prepare_directml(net); net.eval(); identity['hardware']=hardware
        with torch.no_grad():
            ids=tok.encode('Uma planta precisa de água.').ids
            x=torch.tensor([ids],device=device); changed=x.clone(); changed[0,-1]=tok.encode('sol').ids[0]
            p=net(x).detach().cpu(); q=net(changed).detach().cpu()
            identity['causal_prefix_max_difference']=float((p[:,:-1]-q[:,:-1]).abs().max()); assert identity['causal_prefix_max_difference']<1e-4
            for prompt in prompts:
                x=torch.tensor([tok.encode(prompt).ids],device=device); output=[]
                for _ in range(48):
                    value=net(x)[:,-1].argmax(-1,keepdim=True)
                    if value.item()==eos: break
                    output.append(value.item()); x=torch.cat((x,value),1)
                text=tok.decode(output)
                identity['generations'].append({'prompt':prompt,'output':text}); print(json.dumps({'model':name,'prompt':prompt,'output':text},ensure_ascii=True),flush=True)
        report['models'].append(identity); save(); del net
    report['status']='complete'; save()
    print(json.dumps({'status':'complete','tokenizer_identity':True,'token_roundtrip':True,'label_alignment':True,'parameters':102257280,'causal_prefix_tests':True,'weights_updated':False}),flush=True)
if __name__=='__main__': main()
