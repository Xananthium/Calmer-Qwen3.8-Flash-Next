#!/usr/bin/env python3
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_unedited_natural_20261118')
deadline=time.monotonic()+600
while True:
    try:
        state=json.loads((ROOT/'deployment.json').read_text())['state']
        if state=='live':break
        if state.startswith('failed'):raise RuntimeError('Original-model deployment failed; do not run baseline.')
    except (FileNotFoundError,json.JSONDecodeError):pass
    if time.monotonic()>deadline:raise RuntimeError('Original model not ready')
    time.sleep(2)
for script,args,log,scorefile in [('experiment_unedited.py',['loops'],'loop_baseline.log','loop_probe/baseline/scores.json'),
    ('experiment_unedited.py',['baseline'],'baseline.log','baseline/scores.json'),('compare_unedited.py',[],'comparison.log',None)]:
    print(json.dumps(dict(stage=script,args=args,state='starting')),flush=True)
    with (ROOT/log).open('a') as f:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(stage=script,args=args,state='complete')),flush=True)
    if scorefile and not json.loads((ROOT/scorefile).read_text())['complete']:break
