#!/usr/bin/env python3
"""Switch from the original baseline to the prepared partial2/46 experiment."""
import json,subprocess,sys,time
from pathlib import Path
PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT=PARENT/'strata_l2a050_l44_l45_l46a025_20261118'
UNEDITED=PARENT/'strata_unedited_natural_20261118'
sys.path.insert(0,str(UNEDITED))
from load_unedited_baseline import read,write,get,now,STRATA

def main():
    m=read(ROOT/'candidate_manifest.json');assert m['status']=='prepared_not_deployed'
    assert Path(m['target']).stat().st_size==m['source_file_size']
    cfg_path=Path((STRATA/'fn-iq3s/boot.env').read_text().strip().split('=',1)[1]);current=read(cfg_path)
    assert current==read(UNEDITED/'candidate_config.json')
    candidate=read(Path(m['candidate_config']));normalized=json.loads(json.dumps(current))
    for flag in ['--native','--ple-gguf']:normalized['args'][normalized['args'].index(flag)+1]=m['target']
    assert normalized==candidate
    d=dict(state='waiting_for_idle',started_at=now(),from_model='original unedited weights',to_model='44+45 plus layer2 alpha0.50 and layer46 alpha0.25');write(ROOT/'deployment.json',d)
    deadline=time.monotonic()+900
    while True:
        live=get('/metrics')['live']
        if live.get('running',0)==0 and live.get('queued',0)==0:break
        if time.monotonic()>deadline:raise RuntimeError('Production requests have not drained')
        time.sleep(2)
    script=str(STRATA/'scripts/serve-fn-iq3s.sh');d.update(state='restarting',restart_at=now());write(ROOT/'deployment.json',d)
    subprocess.run([script,'stop'],check=True)
    try:
        subprocess.run([script,'start','g4x-l2a050-l44l45-l46a025-natural'],check=True)
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            try:
                health=get('/health')
                if health.get('loaded'):break
            except (OSError,ValueError):pass
            time.sleep(2)
        else:raise RuntimeError('Partial-edit model not ready within300 seconds')
    except BaseException as error:
        subprocess.run([script,'stop'],check=False);subprocess.run([script,'start','g4x-unedited-natural'],check=True)
        d.update(state='failed_restored_unedited_baseline',error=repr(error));write(ROOT/'deployment.json',d)
        raise
    d.update(state='live',ready_at=now(),health=health);write(ROOT/'deployment.json',d)
    m.update(status='deployed_pending_panel');write(ROOT/'candidate_manifest.json',m)
    print(json.dumps(d,indent=2),flush=True)

if __name__=='__main__':main()
