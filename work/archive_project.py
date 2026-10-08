"""Archive this workspace without credentials; large files become release assets."""
from pathlib import Path
import argparse, hashlib, http.client, json, os, re, shutil, subprocess, urllib.request, urllib.error, urllib.parse, zipfile

SOURCE = Path(__file__).resolve().parent.parent
EXPORT = SOURCE / 'archive-export'
REPO = SOURCE.parent / 'cerebro-ia-102m-arquivo'
OWNER = 'anderson10-wp'
NAME = 'cerebro-ia-102m'
TAG = 'arquivo-2026-10-08'
PATTERNS = [r'(?i)KGAT_[A-Za-z0-9]{20,}', r'sk-or-v1-[a-f0-9]{32,}', r'gsk_[A-Za-z0-9]{30,}', r'AQ\.[A-Za-z0-9_-]{30,}', r'AIza[A-Za-z0-9_-]{30,}', r'gh[pousr]_[A-Za-z0-9]{20,}', r'github_pat_[A-Za-z0-9_]{30,}']

def emit(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def clean(text):
    for old in (str(SOURCE), SOURCE.as_posix(), str(SOURCE).replace('\\', '\\\\')):
        text = text.replace(old, '.')
    text = re.sub(re.escape(SOURCE.name), 'cerebro-ia-102m', text, flags=re.I)
    for pattern in PATTERNS: text = re.sub(pattern, '[CREDENCIAL_REMOVIDA]', text)
    text = text.replace('[CREDENCIAL_REMOVIDA]', '[CREDENCIAL_REMOVIDA]')
    return text

README = '''# Cérebro IA — 102M

Projeto pessoal de modelo de linguagem treinado do zero, pausado em 8 de outubro de 2026. Este repositório preserva código, dados, checkpoints, relatórios e tentativas para uma possível retomada. Clonar não inicia treinamento nem chamadas pagas.

## Estado real

O modelo tem 102.257.280 parâmetros: 16 camadas, dimensão 640, 10 cabeças, MLP 1792, contexto 1024 e vocabulário 32.768. Usa RoPE, RMSNorm, SwiGLU e embeddings compartilhados com a saída. O objetivo era conversar em português e, futuramente, ajudar a planejar experimentos com professores externos. Ainda não conversa de maneira confiável nem treina outro modelo autonomamente.

Foram processadas aproximadamente 5,642 bilhões de posições de tokens entre o pré-treino inicial e a continuação na H100; isso inclui repetição do corpus, não 5,642 bilhões de tokens únicos. Os testes pequenos não demonstram qualidade geral.

Referência local: `outputs/local-run-20261007-212517-0aec1e/candidate/best-model.pt`, SHA256 `1ea8da9fa900f7f2ec1beaf5f34511d449ea695732e19d1f7744d4efb21f9a9e`, 8/12 no teste estreito. Pré-treino H100: `outputs/h100-pretrain/result/best-model.pt`, SHA256 `e80f4077b8125f5aaea8eb4a2344e1b0bfa6bf06c4033f27112a6564ecd31a9c`.

Experimentos Kaggle de SFT amplo, raciocínio e linguagem foram concluídos. Houve ganhos em alguns testes, mas falhas de conversa e regressões impediram promoção automática dos candidatos. Os atalhos e estados antigos apontam para versões diferentes; confira o checkpoint antes de usar.

A preparação de `outputs/dialogue-v1/round-03` ficou **incompleta**: professores geraram exemplos, os primeiros 100 receberam revisão e correções manuais, mas o restante não recebeu revisão completa. A construção final do conjunto, revisão independente e novo piloto Kaggle não foram concluídos. Arquivos intermediários e logs documentam falhas, limites e tentativas. Não execute essa etapa como se já estivesse validada.

## Restaurar arquivos grandes

O código e os arquivos menores estão no Git. Modelos, corpus e outros arquivos grandes estão nos assets da release `arquivo-2026-10-08`, deduplicados por SHA256. O manifesto preserva cada caminho original, inclusive cópias idênticas. Um arquivo `.partial` é um download incompleto preservado como tal.

1. Clone este repositório privado.
2. Com GitHub CLI autenticado, execute `gh release download arquivo-2026-10-08 --repo anderson10-wp/cerebro-ia-102m --dir arquivos-release`.
3. Execute `python restore_artifacts.py arquivos-release`.

O restaurador verifica os hashes antes de recriar os caminhos e não sobrescreve arquivos diferentes. Também é possível baixar os assets pela página da release.

## Retomar com cuidado

Use Python 3.12. O ambiente local era Windows com RX 580 8GB, Ryzen 5600G e 16GB RAM. Dependências principais: PyTorch 2.4.1, torch-directml, tokenizers, pyarrow, huggingface_hub e Kaggle CLI. `requirements-archive.txt` registra as versões encontradas no ambiente local; não é uma garantia de compatibilidade em outras máquinas. Não foi incluído o ambiente virtual.

Após instalar as dependências e restaurar os pesos, um exemplo para testar a referência local é:

```powershell
python outputs/h100-pretrain/chat_terminal.py --checkpoint outputs/local-run-20261007-212517-0aec1e/candidate/best-model.pt --device directml
```

Confira `--help` antes de treinar. Wrappers de professores esperam credenciais DPAPI do usuário Windows: elas foram excluídas e precisam ser configuradas novamente. Nenhuma chave de API está incluída. Não há autorização automática para gastar novos créditos.

Experimentos e pesos adicionais podem permanecer no Kaggle:

- https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-sft-treino-oct08
- https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-reasoning-test-oct08
- https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-language-pilot-oct08
- Dataset privado: https://www.kaggle.com/datasets/andersonluan10ad/cerebro-102m-sft-oct08

O arquivamento cobre os arquivos locais existentes, não pesos remotos que nunca foram baixados. Os links permitem localizar esses resultados. Fontes e licenças dos dados estão nos SOURCE-CARDs e manifestos; não há uma licença única atribuída a todo o material de terceiros.

## Conteúdo excluído e alterações do arquivo

Credenciais (`*.dpapi`, `.env`), caches Python e o ambiente virtual foram excluídos. Caminhos absolutos deste workspace e padrões de credenciais nos textos foram substituídos para portabilidade e privacidade. Modelos e dados binários mantêm seus bytes originais. `archive-manifest.json` registra arquivos preservados, exclusões e hashes.
'''

RESTORE = '''from pathlib import Path
import hashlib, json, sys, zipfile, shutil
root=Path(__file__).resolve().parent
assets=Path(sys.argv[1] if len(sys.argv)>1 else 'arquivos-release').resolve()
manifest=json.loads((root/'archive-manifest.json').read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for obj in manifest['objects']:
 archive=assets/obj['asset']
 if sha(archive)!=obj['asset_sha256']:raise SystemExit('Asset corrompido: '+archive.name)
 temp=assets/(obj['sha256']+'.restoring')
 with zipfile.ZipFile(archive) as z, z.open('payload') as src, temp.open('wb') as dst:shutil.copyfileobj(src,dst)
 if sha(temp)!=obj['sha256']:raise SystemExit('Conteúdo corrompido: '+archive.name)
 for name in obj['paths']:
  target=(root/name).resolve()
  if not target.is_relative_to(root):raise SystemExit('Caminho inválido')
  if target.exists():
   if sha(target)!=obj['sha256']:raise SystemExit('Arquivo diferente já existe: '+name)
   continue
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(temp,target)
 temp.unlink();print('Restaurado:',len(obj['paths']),'caminho(s)')
print('Restauração completa.')
'''

def prepare():
    if REPO.exists(): raise SystemExit('Export folder already exists; inspect before rerun.')
    REPO.mkdir(); (EXPORT/'assets').mkdir(parents=True, exist_ok=True)
    manifest={'created':'2026-10-08','status':'paused','files':[],'objects':[],'excluded':[]}
    objects={}
    files=list(SOURCE.glob('*.cmd'))+list(SOURCE.glob('*.md'))+list((SOURCE/'outputs').rglob('*'))+list((SOURCE/'work').rglob('*'))
    for path in files:
        if not path.is_file():continue
        rel=path.relative_to(SOURCE).as_posix()
        if '__pycache__' in path.parts or path.suffix.lower() in ('.dpapi','.pyc') or path.name.startswith('.env'):
            manifest['excluded'].append(rel);continue
        size=path.stat().st_size
        large=path.suffix.lower() in ('.pt','.bin','.partial','.parquet','.zip') or size>10_000_000
        if large:
            digest=sha(path)
            if digest not in objects:
                asset='artifact-'+digest+'.zip'; dest=EXPORT/'assets'/asset
                with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:z.write(path,'payload')
                objects[digest]={'sha256':digest,'size':size,'asset':asset,'asset_size':dest.stat().st_size,'asset_sha256':sha(dest),'paths':[]}
                emit(prepared_asset=asset,bytes=dest.stat().st_size)
            objects[digest]['paths'].append(rel)
            manifest['files'].append({'path':rel,'sha256':digest,'size':size,'storage':'release'})
        else:
            raw=path.read_bytes()
            try: raw=clean(raw.decode('utf-8-sig')).encode('utf-8')
            except UnicodeError:
                if path.suffix.lower() in ('.log','.txt','.cmd','.ps1','.md','.json','.jsonl','.py'):
                    encoding='utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'cp1252'
                    raw=clean(raw.decode(encoding,errors='replace')).encode('utf-8')
            dest=REPO/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            manifest['files'].append({'path':rel,'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'storage':'git'})
    manifest['objects']=list(objects.values())
    (REPO/'archive-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (REPO/'README.md').write_text(README,encoding='utf-8')
    (REPO/'restore_artifacts.py').write_text(RESTORE,encoding='utf-8')
    (REPO/'.gitignore').write_text('.venv*/\n__pycache__/\n*.pyc\n*.dpapi\n.env*\narquivos-release/\n*.pt\n*.bin\n*.partial\n*.parquet\n',encoding='utf-8')
    envpython=SOURCE/'.venv-directml/Scripts/python.exe'
    if envpython.exists():
        p=subprocess.run([str(envpython),'-m','pip','freeze'],capture_output=True,text=True)
        (REPO/'requirements-archive.txt').write_text(clean(p.stdout),encoding='utf-8')
    # Scan every copied text, fail before any publication if a secret remains.
    for p in REPO.rglob('*'):
        if not p.is_file():continue
        try:t=p.read_text(encoding='utf-8')
        except UnicodeError:continue
        if any(re.search(pattern,t) for pattern in PATTERNS):raise SystemExit('Credential pattern remains in '+p.relative_to(REPO).as_posix())
    emit(prepared=True,files=len(manifest['files']),unique_assets=len(objects),asset_bytes=sum(x['asset_size'] for x in objects.values()))

def auth():
    p=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env={**os.environ,'GCM_INTERACTIVE':'never','GIT_TERMINAL_PROMPT':'0'},timeout=30)
    d=dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x)
    if not d.get('password'):raise SystemExit('No cached GitHub authentication')
    return d['password']

def publish():
    token=auth()
    def api(method,path,data=None):
        req=urllib.request.Request('https://api.github.com'+path,data=json.dumps(data).encode() if data is not None else None,method=method,headers={'Authorization':'Bearer '+token,'User-Agent':'project-archive','Accept':'application/vnd.github+json','Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
    user=api('GET','/user')
    if user['login']!=OWNER:raise SystemExit('GitHub account differs from expected owner')
    try: repo=api('GET',f'/repos/{OWNER}/{NAME}')
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        repo=api('POST','/user/repos',{'name':NAME,'private':True,'description':'Arquivo do projeto pessoal de IA 102M, pausado em outubro de 2026.','auto_init':False})
        emit(repository_created=repo['html_url'],private=repo['private'])
    if not repo['private']:raise SystemExit('Existing repository is public; refusing publication')
    if not (REPO/'.git').exists():
        for args in (['init','-b','main'],['config','user.name',user.get('name') or OWNER],['config','user.email',str(user['id'])+'+'+OWNER+'@users.noreply.github.com'],['add','.'],['commit','-m','Preserva projeto 102M, checkpoints e estado de retomada'],['remote','add','origin',repo['clone_url']]):
            subprocess.run(['git',*args],cwd=REPO,check=True,capture_output=True)
    subprocess.run(['git','push','-u','origin','main'],cwd=REPO,check=True,capture_output=True,env={**os.environ,'GCM_INTERACTIVE':'never','GIT_TERMINAL_PROMPT':'0'})
    emit(code_pushed=True)
    try: release=api('GET',f'/repos/{OWNER}/{NAME}/releases/tags/{TAG}')
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        release=api('POST',f'/repos/{OWNER}/{NAME}/releases',{'tag_name':TAG,'target_commitish':'main','name':'Arquivo completo local — 8 de outubro de 2026','body':'Projeto pausado. Código e instruções no repositório; checkpoints e dados locais nos assets abaixo. Use archive-manifest.json e restore_artifacts.py para restaurar cada caminho e verificar SHA256. Credenciais, ambiente virtual e caches foram excluídos. Pesos remotos nunca baixados continuam nos links Kaggle registrados no README.','draft':True})
    existing={x['name']:x for x in api('GET',f"/repos/{OWNER}/{NAME}/releases/{release['id']}/assets?per_page=100")}
    manifest=json.loads((REPO/'archive-manifest.json').read_text(encoding='utf-8'))
    for obj in manifest['objects']:
        name=obj['asset'];path=EXPORT/'assets'/name
        if name in existing:
            if existing[name]['size']!=path.stat().st_size:raise SystemExit('Existing asset size mismatch')
            continue
        url=urllib.parse.urlsplit(release['upload_url'].split('{')[0]+'?name='+name)
        connection=http.client.HTTPSConnection(url.netloc,timeout=180)
        connection.putrequest('POST',url.path+'?'+url.query)
        for k,v in {'Authorization':'Bearer '+token,'User-Agent':'project-archive','Content-Type':'application/zip','Content-Length':str(path.stat().st_size)}.items():connection.putheader(k,v)
        connection.endheaders()
        emit(uploading=name,bytes=path.stat().st_size)
        with path.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):connection.send(b)
        response=connection.getresponse();raw=response.read();connection.close()
        if response.status not in (200,201):raise SystemExit('Asset upload failed HTTP '+str(response.status))
        asset=json.loads(raw)
        if asset['size']!=obj['asset_size']:raise SystemExit('Uploaded asset size mismatch')
        if asset.get('digest') and asset['digest']!='sha256:'+obj['asset_sha256']:raise SystemExit('Uploaded asset SHA mismatch')
        emit(uploaded=name)
    assets=api('GET',f"/repos/{OWNER}/{NAME}/releases/{release['id']}/assets?per_page=100")
    byname={x['name']:x for x in assets}
    for obj in manifest['objects']:
        asset=byname.get(obj['asset'])
        if not asset or asset['state']!='uploaded' or asset['size']!=obj['asset_size']:raise SystemExit('Remote archive incomplete')
        if asset.get('digest') and asset['digest']!='sha256:'+obj['asset_sha256']:raise SystemExit('Remote SHA mismatch')
    release=api('PATCH',f"/repos/{OWNER}/{NAME}/releases/{release['id']}",{'draft':False})
    result={'repository':repo['html_url'],'release':release['html_url'],'private':True,'verified_assets':len(manifest['objects']),'preserved_files':len(manifest['files'])}
    (EXPORT/'publication.json').write_text(json.dumps(result,indent=2),encoding='utf-8');emit(**result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','publish']);args=parser.parse_args()
    prepare() if args.action=='prepare' else publish()
