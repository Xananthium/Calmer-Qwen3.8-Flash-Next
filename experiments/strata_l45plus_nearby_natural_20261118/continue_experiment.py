#!/usr/bin/env python3
"""Run the already authorized paired phases after the current baseline drains."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45plus_nearby_natural_20261118')

def run(script,operation,log):
    print(json.dumps({'stage':script,'operation':operation,'state':'starting'}),flush=True)
    with (ROOT/log).open('a') as output:
        subprocess.run([sys.executable,'-u',str(ROOT/script),*([operation] if operation else [])],stdout=output,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'stage':script,'operation':operation,'state':'complete'}),flush=True)

deadline=time.monotonic()+1800
while True:
    scores=json.loads((ROOT/'baseline/scores.json').read_text())
    if scores['complete'] and (ROOT/'baseline/identity_after.json').exists():break
    if time.monotonic()>deadline:raise RuntimeError('Baseline did not complete within 30 minutes')
    time.sleep(5)
run('loop_panel.py','baseline','loop_baseline.log')
run('deploy_l46_addition.py',None,'deployment.log')
run('loop_panel.py','edited','loop_edited.log')
run('natural_panel.py','edited','edited.log')
run('compare_natural.py',None,'comparison.log')
