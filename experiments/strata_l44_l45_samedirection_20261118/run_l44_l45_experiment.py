#!/usr/bin/env python3
"""Wait for verified model preparation, then deploy and evaluate the chosen pair."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l44_l45_samedirection_20261118')

def run(script,args,log):
    print(json.dumps({'stage':script,'args':args,'state':'starting'}),flush=True)
    with (ROOT/log).open('a') as f:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'stage':script,'args':args,'state':'complete'}),flush=True)

deadline=time.monotonic()+1800
while True:
    try:
        if json.loads((ROOT/'candidate_manifest.json').read_text())['status']=='prepared_not_deployed':break
    except (FileNotFoundError,json.JSONDecodeError):pass
    if time.monotonic()>deadline:raise RuntimeError('Preparation did not finish')
    time.sleep(5)
run('deploy_l44_l45.py',[],'deployment.log')
run('experiment_l44_l45.py',['loops'],'loop_edited.log')
if not json.loads((ROOT/'loop_probe/edited/scores.json').read_text())['complete']:sys.exit(0)
run('experiment_l44_l45.py',['edited'],'edited.log')
if not json.loads((ROOT/'edited/scores.json').read_text())['complete']:sys.exit(0)
run('compare_l44_l45.py',[],'comparison.log')
