import argparse
import json
import os
from pathlib import Path
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--prompt', default='Responda em português: quanto é 7 vezes 8?')
p.add_argument('--endpoint', required=True)
p.add_argument('--output')
args = p.parse_args()
body = json.dumps({'model': 'gpt-oss-20b', 'messages': [{'role': 'user', 'content': args.prompt}],
                   'max_tokens': 1536, 'temperature': 0, 'reasoning_effort': 'low'}).encode()
request = urllib.request.Request(args.endpoint.rstrip('/') + '/chat/completions', data=body,
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['GPT_OSS_LOCAL_KEY']})
with urllib.request.urlopen(request, timeout=180) as response:
    data = json.load(response)
result = {'model': data.get('model'), 'answer': data['choices'][0]['message'].get('content'),
          'finish_reason': data['choices'][0].get('finish_reason'), 'usage': data.get('usage')}
if args.output:
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=True))
