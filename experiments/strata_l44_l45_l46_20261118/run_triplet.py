#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l44_l45_l46_20261118')
deadline=time.monotonic()+600
while True:
    try:
        if json.loads((ROOT/'deployment.json').read_text())['state']=='live':break
    except (FileNotFoundError,json.JSONDecodeError):pass
    if time.monotonic()>deadline:raise RuntimeError('Triplet not ready')
    time.sleep(2)
for script,args,log,scorefile in [('experiment_triplet.py',['loops'],'loop_edited.log','loop_probe/edited/scores.json'),
                                ('experiment_triplet.py',['edited'],'edited.log','edited/scores.json'),
                                ('compare_triplet.py',[],'comparison.log',None)]:
    print(json.dumps({'stage':script,'args':args,'state':'starting'}),flush=True)
    with (ROOT/log).open('a') as f:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'stage':script,'args':args,'state':'complete'}),flush=True)
    if scorefile and not json.loads((ROOT/scorefile).read_text())['complete']:break
