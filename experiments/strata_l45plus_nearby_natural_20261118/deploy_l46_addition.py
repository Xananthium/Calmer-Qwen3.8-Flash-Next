#!/usr/bin/env python3
"""Load the cumulative edit after the natural-completion baseline is captured."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45plus_nearby_natural_20261118')
STRATA=Path('/home/researcher/strata-122b')

def read(p):return json.loads(p.read_text())

def get(path):
    with urlopen('http://127.0.0.1:8001'+path,timeout=5) as response:return json.load(response)

def main():
    scores=read(ROOT/'baseline/scores.json')
    assert scores['complete'] and len(scores['rows'])==6
    loops=read(ROOT/'loop_probe/baseline/scores.json')
    assert loops['complete'] and len(loops['rows'])==8
    before=read(ROOT/'baseline/identity_before.json')
    after=read(ROOT/'baseline/identity_after.json')
    assert before['engines']==after['engines'] and before['config']==after['config']
    loop_after=read(ROOT/'loop_probe/baseline/identity_after.json')
    assert after['engines']==loop_after['engines'] and after['config']==loop_after['config']
    m=read(ROOT/'candidate_manifest.json')
    assert m['status']=='prepared_not_deployed' and m['all_unchanged_bytes_verified_against_parent']
    config=read(Path(m['candidate_config']))
    normalized=json.loads(json.dumps(config))
    for flag in ['--native','--ple-gguf']:
        index=normalized['args'].index(flag)+1
        assert normalized['args'][index]==m['target']
        normalized['args'][index]=m['source']
    assert normalized==after['config']
    current_cfg=Path((STRATA/'fn-iq3s/boot.env').read_text().strip().split('=',1)[1])
    assert read(current_cfg)==after['config']
    deadline=time.monotonic()+900
    while True:
        live=get('/metrics')['live']
        if live.get('running',0)==0 and live.get('queued',0)==0:break
        if time.monotonic()>=deadline:
            raise RuntimeError('Production requests have not drained; preserve them and retry deployment later.')
        time.sleep(2)
    deployment={'started_at':datetime.now(timezone.utc).isoformat(),'from':'g4x-l45svd1-natural',
                'to':'g4x-l45l46svd1-natural','state':'restarting','baseline_complete':True}
    (ROOT/'deployment.json').write_text(json.dumps(deployment,indent=2)+'\n')
    script=str(STRATA/'scripts/serve-fn-iq3s.sh')
    subprocess.run([script,'stop'],check=True)
    subprocess.run([script,'start','g4x-l45l46svd1-natural'],check=True)
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        try:
            h=get('/health')
            if h.get('loaded'):
                sys.path.insert(0,str(ROOT))
                import natural_panel as n
                n.harness.ROOT=ROOT;n.harness.SOURCE=n.SOURCE;n.harness.TARGET=n.TARGET
                identity=n.identity('edited')
                (ROOT/'live_edited_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
                deployment.update(state='live',ready_at=datetime.now(timezone.utc).isoformat(),
                                  engine_pid=identity['engines'][0]['pid'],health=h)
                (ROOT/'deployment.json').write_text(json.dumps(deployment,indent=2)+'\n')
                m.update(status='deployed_pending_natural_panel',deployment=str(ROOT/'deployment.json'))
                (ROOT/'candidate_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
                print(json.dumps(deployment),flush=True)
                return
        except (OSError,ValueError):pass
        time.sleep(2)
    raise RuntimeError('Cumulative candidate did not become ready in 240 seconds')

if __name__=='__main__':main()
