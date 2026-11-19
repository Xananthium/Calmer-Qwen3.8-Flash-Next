#!/usr/bin/env python3
"""Restore one public request template using an authorized local messages JSON file."""
import argparse,hashlib,json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--template',required=True,type=Path)
p.add_argument('--messages',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
request=json.loads(a.template.read_text())
messages=json.loads(a.messages.read_text())
frozen=request.pop('messages_withheld')
data=json.dumps(messages,sort_keys=True,separators=(',',':')).encode()
if hashlib.sha256(data).hexdigest()!=frozen['sha256']:
    raise ValueError('Messages differ from the frozen request; no output written.')
request['messages']=messages
with a.output.open('x') as f: json.dump(request,f,indent=2,ensure_ascii=False); f.write('\n')
print('Restored matching messages. This tool makes no inference request.')
