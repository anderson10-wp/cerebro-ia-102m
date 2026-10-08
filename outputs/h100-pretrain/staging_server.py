"""Private CPU-only data staging. Credentials and signed URLs never touch disk/logs."""
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import hashlib,json,os,threading,urllib.parse,traceback
import requests,torch
os.environ['LIGHTNING_DISABLE_VERSION_CHECK']='1'
from lightning_sdk import Studio,Teamspace
from lightning_sdk.api.utils import cached_lightning_client
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'data-manifest.json').read_text())
status={'status':'ready','gpu_used':False,'completed_files':[]}
busy=False
class StageError(Exception): pass

def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(body,key,user):
    previous={k:os.environ.get(k) for k in ('LIGHTNING_API_KEY','LIGHTNING_USER_ID')}
    try:
        os.environ['LIGHTNING_API_KEY']=key; os.environ['LIGHTNING_USER_ID']=user
        client=cached_lightning_client()
        account=client.auth_service_get_user()
        if account.username!='anderson10-wp': raise StageError('Unexpected account')
        team=Teamspace('default-project',user=account.username)
        target=Studio('cerebro-102m-h100-continuation-nebius',teamspace=team,create_ok=False,disable_secrets=True)
        if target.id!=body['target_id'] or str(target.status)!='Stopped': raise StageError('Expected stopped target')
        expected={f'{d}-{s}.bin':(manifest['counts'][d][s]*2,manifest['sha256'][f'{d}-{s}.bin']) for d in ('pt','en','code') for s in ('train','validation')}
        expected['source.pt']=(None,None)
        if len(body['files'])!=len(expected) or {f['name'] for f in body['files']}!=set(expected): raise StageError('Unexpected artifact list')
        inputs=ROOT/'inputs'; inputs.mkdir(exist_ok=True)
        def fetch(item):
            name=item['name']; parsed=urllib.parse.urlparse(item['url'])
            if parsed.scheme!='https' or not ((parsed.hostname=='www.kaggleusercontent.com' and parsed.path.startswith('/kf/')) or (parsed.hostname=='storage.googleapis.com' and parsed.path.startswith('/kaggle-kernels-output/'))):
                raise StageError('Unexpected artifact host or namespace')
            size,wanted_hash=expected[name]; path=inputs/name; count=0
            if size and path.is_file() and path.stat().st_size==size and digest(path)==wanted_hash:
                return {'name':name,'bytes':size,'sha256':wanted_hash}
            with requests.get(item['url'],stream=True,timeout=(30,90),allow_redirects=False) as response:
                if response.status_code!=200: raise StageError('Artifact download failed')
                with path.open('wb') as f:
                    for block in response.iter_content(8*1024**2):
                        count+=len(block)
                        if count>(size if size else 1_500_000_000): raise StageError('Artifact exceeds expected size')
                        f.write(block)
            if size and count!=size: raise StageError('Incomplete corpus artifact')
            sha=digest(path)
            if wanted_hash and sha!=wanted_hash: raise StageError('Corpus hash mismatch')
            return {'name':name,'bytes':count,'sha256':sha}
        status['status']='downloading'
        with ThreadPoolExecutor(max_workers=3) as pool:
            records=list(pool.map(fetch,body['files']))
        source=torch.load(inputs/'source.pt',map_location='cpu',weights_only=True)
        if source['tokens']!=1_999_962_112 or source['corpus_hash']!=digest(ROOT/'data-manifest.json') or source['tokenizer_hash']!=digest(ROOT/'tokenizer.json'):
            raise StageError('Wrong source checkpoint or tokenizer')
        if sum(source['counts'].values())*1024!=source['tokens']: raise StageError('Wrong training cursors')
        if not all(torch.isfinite(v).all().item() for v in source['model'].values()): raise StageError('Source contains non-finite weights')
        del source
        status['status']='uploading'
        for name,remote in [('model.py','model.py'),('training_helpers.py','training_helpers.py'),('train_h100.py','train_h100.py'),('data-manifest.json','data/manifest.json'),('tokenizer.json','data/tokenizer.json')]:
            target.upload_file(str(ROOT/name),remote_path=remote,progress_bar=False)
        for record in records:
            remote='source.pt' if record['name']=='source.pt' else 'data/'+record['name']
            target.upload_file(str(inputs/record['name']),remote_path=remote,progress_bar=False)
            status['completed_files'].append(record)
        status.update(status='prepared',target_id=target.id,credentials_persisted=False)
    except Exception as error:
        frame=traceback.extract_tb(error.__traceback__)[-1]
        status.update(status='failed',error_type=type(error).__name__,error_function=frame.name,error_line=frame.lineno)
        if isinstance(error,StageError): status['error_code']=str(error)
    finally:
        for k,v in previous.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v
        (ROOT/'staging-report.json').write_text(json.dumps(status,indent=2),encoding='utf-8')

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_): pass
    def reply(self,code,value):
        data=json.dumps(value).encode(); self.send_response(code); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        self.reply(200,status)
    def do_POST(self):
        global busy
        size=int(self.headers.get('Content-Length','0'))
        if self.path!='/prepare' or busy or not 0<size<40000: self.reply(400,{'error':'invalid_request'}); return
        key=self.headers.get('X-Training-Api-Key'); user=self.headers.get('X-Training-User-ID')
        if not key or not user: self.reply(401,{'error':'missing_auth'}); return
        try: body=json.loads(self.rfile.read(size))
        except ValueError: self.reply(400,{'error':'invalid_json'}); return
        busy=True
        threading.Thread(target=prepare,args=(body,key,user),daemon=True).start()
        self.reply(202,{'status':'accepted'})

if __name__=='__main__':
    ThreadingHTTPServer(('0.0.0.0',8062),Handler).serve_forever()

