#!/usr/bin/env python3
"""Honor the requested early stop after the in-flight math request completes."""
import json
import os
from pathlib import Path
import signal
import time
import natural_panel as n

r=n.ROOT
deadline=time.monotonic()+1800
while True:
    try:
        result=json.loads((r/'edited/05.result.json').read_text())
        break
    except (FileNotFoundError,json.JSONDecodeError):
        if time.monotonic()>deadline:raise RuntimeError('Current math result did not arrive')
        time.sleep(0.05)
killed=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        args=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        if len(args)>=3 and args[-2:]==[str(r/'natural_panel.py'),'edited']:
            os.kill(int(p.name),signal.SIGTERM);killed.append(int(p.name))
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
protocol=json.loads((r/'protocol.json').read_text())
rows=[]
for case in protocol['cases']:
    p=r/'edited'/(case['case']+'.result.json')
    if p.exists():rows.append(n.score(case,json.loads(p.read_text())))
n.harness.write(r/'edited/scores.json',dict(arm='edited',protocol_sha256=n.harness.digest((r/'protocol.json').read_bytes()),
    rows=rows,complete=False,stopped_by_user=True,planned_cases=[c['case'] for c in protocol['cases']],
    reason='User requested dropping layer46 after the current math prompt and moving to layer44 plus layer45.'))
n.harness.ROOT=r;n.harness.SOURCE=n.SOURCE;n.harness.TARGET=n.TARGET
after=n.identity('edited');n.harness.write(r/'edited/identity_after_early_stop.json',after)
decision=dict(at=n.harness.now(),stopped_client_pids=killed,finished_cases=[row['case'] for row in rows],
    current_math=n.score(next(c for c in protocol['cases'] if c['case']=='05'),result),
    next_model='Layer44 plus Layer45, built from Layer45-only artifact, with Layer46 original weights restored.',
    interpretation='Partial exploratory panel stopped by user; do not report six-case completion or fill unrun cases.',
    next_started_request_may_be_cancelled='The runner can dispatch the next request before its result file is observed; any such stream is kept as an administrative interruption, not scored.')
n.harness.write(r/'ended_early_by_user.json',decision)
m=json.loads((r/'candidate_manifest.json').read_text());m['status']='evaluation_stopped_by_user_pending_l44_replacement';m['early_stop']=str(r/'ended_early_by_user.json');n.harness.write(r/'candidate_manifest.json',m)
print(json.dumps(decision,indent=2),flush=True)
