"""Stage directly from Kaggle to Lightning CPU over an authenticated HTTPS endpoint."""
import json,os,sys,time
from pathlib import Path
import requests
os.environ['LIGHTNING_DISABLE_VERSION_CHECK']='1'
from lightning_sdk import Studio,Teamspace,Machine
from lightning_sdk.machine import CloudProvider
from lightning_sdk.api.utils import cached_lightning_client
from lightning_sdk.lightning_cloud.openapi import EndpointServiceCreateEndpointBody,V1EndpointAuth,V1UpstreamCloudSpace,V1EndpointType
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
ROOT=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')

def main():
    client=cached_lightning_client(); team=Teamspace('default-project',user=client.auth_service_get_user().username)
    target=Studio('cerebro-102m-h100-continuation-nebius',teamspace=team,cloud=CloudProvider.NEBIUS,disable_secrets=True)
    if target.cloud_account!='lightning-nebius-prod' or str(target.status)!='Stopped': raise RuntimeError('Unexpected target')
    state=json.loads((ROOT/'cloud-state.json').read_text())
    state.update(studio_id=target.id,provider=target.cloud_account,status='cpu_staging')
    (ROOT/'cloud-state.json').write_text(json.dumps(state,indent=2))
    cpu=Studio('cerebro-102m-h100-staging',teamspace=team,cloud=CloudProvider.AWS,disable_secrets=True)
    cpu.show_progress=False
    endpoint=None
    try:
        if str(cpu.status)=='Stopped': cpu.start(Machine.CPU)
        live=cpu._studio_api.get_studio_status(cpu.id,team.id).in_use
        if not live.free or live.price!=0: raise RuntimeError('Staging CPU must be free')
        files={'staging_server.py':'staging_server.py','model.py':'model.py','training_helpers.py':'training_helpers.py','train_h100.py':'train_h100.py',
               'data/corpus/manifest.json':'data-manifest.json','data/corpus/tokenizer.json':'tokenizer.json'}
        for local,remote in files.items(): cpu.upload_file(str(ROOT/local),remote_path=remote,progress_bar=False)
        endpoints=[e for e in cpu.list_ports() if e.name=='private-staging-secure' and e.cloudspace and e.ports]
        endpoint=endpoints[0] if endpoints else client.endpoint_service_create_endpoint(project_id=team.id,
            body=EndpointServiceCreateEndpointBody(name='private-staging-secure',ports=['8062'],
                auth=V1EndpointAuth(enabled=True,user_api_key=True),
                cloudspace=V1UpstreamCloudSpace(cloudspace_id=cpu.id,port='8062',type=V1EndpointType.PLUGIN_PORT,auto_start=False)))
        if not endpoint.auth.enabled or not endpoint.auth.user_api_key: raise RuntimeError('Staging endpoint must require user API key')
        for _ in range(20):
            matches=[e for e in cpu.list_ports() if e.id==endpoint.id]
            if matches: endpoint=matches[0]
            if endpoint.urls: break
            time.sleep(2)
        else: raise RuntimeError('Private endpoint URL unavailable')
        url=endpoint.urls[0].rstrip('/')
        if not url.startswith('https://'): raise RuntimeError('HTTPS is required')
        print(json.dumps({'event':'free_cpu_ready','gpu_started':False,'endpoint_auth':True}),flush=True)
        cpu.run('nohup timeout --signal=TERM --kill-after=20s 900s python -u staging_server.py > staging-server.log 2>&1 < /dev/null &')
        api=KaggleApi(); api.authenticate(); artifacts=[]
        for slug,names in [('cerebro-v1-102m-corpus',{f'{d}-{s}.bin' for d in ('pt','en','code') for s in ('train','validation')}),('cerebro-v1-102m-train-03',{'last.pt'})]:
            request=ApiListKernelSessionOutputRequest(); request.user_name='andersonluan10ad'; request.kernel_slug=slug; request.page_size=200
            with api.build_kaggle_client() as kaggle: response=kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
            for f in response.files:
                if Path(f.file_name).name in names: artifacts.append({'name':'source.pt' if Path(f.file_name).name=='last.pt' else Path(f.file_name).name,'url':f.url})
        headers={'Authorization':'Bearer '+os.environ['LIGHTNING_API_KEY'],
                 'X-Training-Api-Key':os.environ['LIGHTNING_API_KEY'],'X-Training-User-ID':os.environ['LIGHTNING_USER_ID']}
        for _ in range(20):
            try:
                response=requests.get(url,headers=headers,timeout=20)
                if response.status_code==200: break
            except requests.RequestException: pass
            time.sleep(2)
        else: raise RuntimeError('Private CPU endpoint did not become ready')
        response=requests.post(url+'/prepare',headers=headers,json={'target_id':target.id,'files':artifacts},timeout=30)
        if response.status_code!=202: raise RuntimeError('Staging request rejected')
        last=None
        for _ in range(40):
            time.sleep(15)
            response=requests.get(url,headers=headers,timeout=30)
            if response.status_code!=200: raise RuntimeError('Staging status unavailable')
            progress=response.json()
            current=(progress['status'],len(progress['completed_files']))
            if current!=last: print(json.dumps(progress),flush=True); last=current
            if progress['status']=='failed': raise RuntimeError('CPU staging failed: '+progress.get('error_type','unknown'))
            if progress['status']=='prepared':
                state.update(status='prepared',staging_report=progress)
                (ROOT/'cloud-state.json').write_text(json.dumps(state,indent=2))
                (ROOT/'staging-report.json').write_text(json.dumps(progress,indent=2))
                return
        raise TimeoutError('CPU staging deadline exceeded')
    finally:
        if str(cpu.status) in ('Running','Pending'): cpu.stop()
        print(json.dumps({'event':'cpu_stopped','status':str(cpu.status)}),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as error:
        import traceback
        frame=traceback.extract_tb(error.__traceback__)[-1]
        print(json.dumps({'error_type':type(error).__name__,'function':frame.name,'line':frame.lineno}),flush=True)
        sys.exit(1)
