import json,os,urllib.request,urllib.error
secret=os.environ['GEMINI_API_KEY']
try:
    request=urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models',headers={'x-goog-api-key':secret})
    with urllib.request.urlopen(request,timeout=30) as response: result=json.load(response)
    print(json.dumps([{'name':r['name'],'methods':r.get('supportedGenerationMethods',[])} for r in result.get('models',[]) if 'flash' in r['name'].lower()],ensure_ascii=False))
except urllib.error.HTTPError as error:
    print(json.dumps({'http_status':error.code,'message':str(json.loads(error.read()).get('error',{}).get('message','')).replace(secret,'[REDACTED]')[:500]}))
