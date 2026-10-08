"""Chat/test terminal for our local 102M base model. No network or paid APIs."""
import argparse,sys
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Brain,Config

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='Teste local da nossa IA')
    parser.add_argument('--prompt',help='Executa uma mensagem e encerra')
    parser.add_argument('--texto',action='store_true',help='Completa o texto diretamente')
    parser.add_argument('--tokens',type=int,default=64)
    parser.add_argument('--temperatura',type=float,default=.7)
    parser.add_argument('--historico',action='store_true',help='Inclui mensagens anteriores no contexto')
    parser.add_argument('--device',choices=['cpu','directml'],default='cpu')
    parser.add_argument('--checkpoint',help='Checkpoint alternativo em vez do modelo oficial do chat')
    args=parser.parse_args()
    if not 1<=args.tokens<=512 or args.temperatura<0: parser.error('Tokens: 1 a 512; temperatura: zero ou maior')
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4); torch.manual_seed(42)
    checkpoint=Path(args.checkpoint) if args.checkpoint else ROOT/'result/best-model.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=Brain(Config(**saved['config'])); net.load_state_dict(saved['model']); net.eval()
    del saved
    device=torch.device('cpu'); hardware='processador'
    if args.device=='directml':
        from directml_backend import prepare_directml
        device,hardware=prepare_directml(net)
    tok=Tokenizer.from_file(str(ROOT/'result/tokenizer.json'))
    eos=tok.token_to_id('<|eos|>')
    history=''
    @torch.no_grad()
    def answer(message,raw=False):
        nonlocal history
        prompt=message if raw else (history if args.historico else '')+'Pergunta: '+message+'\nResposta:'
        ids=tok.encode(prompt).ids
        if not ids: return ''
        ids=ids[-(net.cfg.context-args.tokens):]
        x=torch.tensor([ids],device=device); generated=[]
        for _ in range(args.tokens):
            logits=net(x[:,-net.cfg.context:])[:,-1].float().cpu()
            if args.temperatura==0:
                next_id=logits.argmax(-1,keepdim=True)
            else:
                logits=logits/args.temperatura
                cutoff=logits.topk(40).values[:,-1:]
                logits=logits.masked_fill(logits<cutoff,float('-inf'))
                next_id=torch.multinomial(logits.softmax(-1),1)
            value=next_id.item()
            if value==eos: break
            generated.append(value); x=torch.cat([x,next_id.to(device)],dim=1)
            text=tok.decode(generated)
            if not raw and '\nPergunta:' in text: break
        text=tok.decode(generated)
        if not raw:
            text=text.split('\nPergunta:',1)[0].strip()
            if args.historico:
                history=tok.decode(tok.encode(prompt+' '+text+'\n').ids[-(net.cfg.context//2):])
        return text
    if args.prompt is not None:
        print(answer(args.prompt,args.texto)); return
    print('Nossa IA — 102M parâmetros, checkpoint: '+checkpoint.parent.name+'. Executando em: '+hardware+'.')
    print('Modelo-base: as respostas ainda podem ser incorretas ou fora do assunto.')
    print('Digite /sair para fechar, /limpar para limpar o contexto ou /texto seguido de um texto para completar.')
    if not args.historico: print('Cada mensagem é um teste independente. Use --historico para incluir a conversa anterior.')
    while True:
        try:
            message=input('\nVocê: ').strip()
            if message.lower() in ('/sair','sair','exit'): break
            if message=='/limpar': history=''; print('Contexto limpo.'); continue
            if not message: continue
            raw=message.startswith('/texto ')
            result=answer(message[7:] if raw else message,raw)
            print('Ela: '+(result or '[Fim da geração sem texto.]'))
        except EOFError: break
        except KeyboardInterrupt: print('\nEncerrado.'); break

if __name__=='__main__': main()
