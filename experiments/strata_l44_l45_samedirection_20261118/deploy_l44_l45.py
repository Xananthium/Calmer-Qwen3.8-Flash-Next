#!/usr/bin/env python3
"""Replace the stopped layer46 experiment with the authorized layer44+45 model."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
import experiment_l44_l45 as experiment

ROOT=experiment.ROOT;PREVIOUS=experiment.PREVIOUS
STRATA=Path('/home/researcher/strata-122b')
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def get(p):
    with urlopen('http://127.0.0.1:8003'+p,timeout=10) as r:return json.load(r)

def main():
    m=read(ROOT/'candidate_manifest.json')
    assert m['status']=='prepared_not_deployed'
    assert m.get('all_unchanged_bytes_verified_against_parent') or m.get('targeted_integrity_verified')
    assert (PREVIOUS/'ended_early_by_user.json').exists()
    base=read(PREVIOUS/'baseline/scores.json');assert base['complete'] and len(base['rows'])==6
    loops=read(PREVIOUS/'loop_probe/baseline/scores.json');assert loops['complete'] and len(loops['rows'])==8
    expected=read(ROOT/'baseline_natural_config.json')
    for c in [read(Path(m['candidate_config'])),read(Path((STRATA/'fn-iq3s/boot.env').read_text().strip().split('=',1)[1]))]:
        for flag in ['--native','--ple-gguf']:
            i=c['args'].index(flag)+1
            assert c['args'][i] in [m['target'],read(PREVIOUS/'candidate_manifest.json')['target']]
            c['args'][i]=m['source']
        assert c==expected
    deadline=time.monotonic()+900
    while True:
        live=get('/metrics')['live']
        if live.get('running',0)==0 and live.get('queued',0)==0:break
        if time.monotonic()>deadline:raise RuntimeError('Requests did not drain; do not interrupt user traffic')
        time.sleep(2)
    d=dict(started_at=datetime.now(timezone.utc).isoformat(),state='restarting',from_model='Layer45+46',to_model='Layer44+45',layer46='original weights restored')
    write(ROOT/'deployment.json',d)
    script=str(STRATA/'scripts/serve-fn-iq3s.sh')
    subprocess.run([script,'stop'],check=True);subprocess.run([script,'start','g4x-l44l45-natural'],check=True)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        try:
            h=get('/health')
            if h.get('loaded'):
                experiment.configure();identity=experiment.identity('edited')
                write(ROOT/'live_edited_identity.json',identity)
                d.update(state='live',ready_at=datetime.now(timezone.utc).isoformat(),health=h,engine_pid=identity['engines'][0]['pid'])
                write(ROOT/'deployment.json',d)
                m.update(status='deployed_pending_panel',deployment=str(ROOT/'deployment.json'));write(ROOT/'candidate_manifest.json',m)
                old=read(PREVIOUS/'candidate_manifest.json');old['status']='preserved_not_serving_after_user_selected_l44';write(PREVIOUS/'candidate_manifest.json',old)
                print(json.dumps(d,indent=2),flush=True);return
        except (OSError,ValueError):pass
        time.sleep(2)
    raise RuntimeError('Candidate did not become ready within 300 seconds')

if __name__=='__main__':main()
