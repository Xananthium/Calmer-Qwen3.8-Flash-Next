#!/usr/bin/env python3
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l2a050_l44_l45_l46a025_20261118')
deadline=time.monotonic()+600
while True:
    try:
        state=json.loads((ROOT/'deployment.json').read_text())['state']
        if state=='live':break
        if state.startswith('failed'):raise RuntimeError('Deployment failed; do not run partial-edit tests.')
    except (FileNotFoundError,json.JSONDecodeError):pass
    if time.monotonic()>deadline:raise RuntimeError('Partial-edit model not ready')
    time.sleep(2)
for script,args,log,scorefile in [('experiment_partial_2_46.py',['loops'],'loop_edited.log','loop_probe/edited/scores.json'),
    ('experiment_partial_2_46.py',['edited'],'edited.log','edited/scores.json'),('compare_partial_2_46.py',[],'comparison.log',None)]:
    print(json.dumps(dict(stage=script,args=args,state='starting')),flush=True)
    with (ROOT/log).open('a') as f:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(stage=script,args=args,state='complete')),flush=True)
    if scorefile and not json.loads((ROOT/scorefile).read_text())['complete']:break
