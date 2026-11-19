#!/usr/bin/env python3
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l2_l44_l45_20261118')
deadline=time.monotonic()+600
while True:
    try:
        state=json.loads((ROOT/'deployment.json').read_text())['state']
        if state=='live':break
        if state.startswith('failed'):raise RuntimeError('Deployment failed; do not run candidate tests.')
    except (FileNotFoundError,json.JSONDecodeError):pass
    if time.monotonic()>deadline:raise RuntimeError('Layer2 candidate not ready')
    time.sleep(2)
for script,args,log,scorefile in [('experiment_layer2.py',['loops'],'loop_edited.log','loop_probe/edited/scores.json'),
    ('experiment_layer2.py',['edited'],'edited.log','edited/scores.json'),('compare_layer2.py',[],'comparison.log',None)]:
    print(json.dumps(dict(stage=script,args=args,state='starting')),flush=True)
    with (ROOT/log).open('a') as f:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(stage=script,args=args,state='complete')),flush=True)
    if scorefile and not json.loads((ROOT/scorefile).read_text())['complete']:break
