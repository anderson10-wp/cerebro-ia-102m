import json,urllib.parse,requests
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
api=KaggleApi(); api.authenticate()
for slug in ('cerebro-v1-102m-corpus','cerebro-v1-102m-train-03'):
    request=ApiListKernelSessionOutputRequest(); request.user_name='andersonluan10ad'; request.kernel_slug=slug; request.page_size=200
    with api.build_kaggle_client() as client: response=client.kernels.kernels_api_client.list_kernel_session_output(request)
    for f in response.files:
        if f.file_name.endswith(('pt-train.bin','last.pt')):
            parsed=urllib.parse.urlparse(f.url)
            print(json.dumps({'artifact':f.file_name,'scheme':parsed.scheme,'hostname':parsed.hostname,'namespace':parsed.path.split('/')[1]}))
            with requests.get(f.url,headers={'Range':'bytes=0-0'},stream=True,timeout=30) as result:
                print(json.dumps({'status':result.status_code,'hosts':[urllib.parse.urlparse(r.url).hostname for r in [*result.history,result]],'namespace':urllib.parse.urlparse(result.url).path.split('/')[1]}))
