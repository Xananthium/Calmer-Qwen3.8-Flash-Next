#!/usr/bin/env python3
"""Restore layer46 and patch layer2 only while the working model is stopped."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
from pathlib import Path
from urllib.request import urlopen

PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT=PARENT/'strata_l2_l44_l45_20261118'
STRATA=Path('/home/researcher/strata-122b')
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def get(p):
    with urlopen('http://127.0.0.1:8003'+p,timeout=10) as response:return json.load(response)

def main():
    m=read(ROOT/'candidate_manifest.json');source=Path(m['source']);target=Path(m['target']);baseline=Path(m['baseline_model'])
    assert m['status']=='prepared_not_deployed' and source.exists() and not target.exists() and not baseline.exists()
    target.parent.mkdir(exist_ok=True);baseline.parent.mkdir(exist_ok=True)
    assert source.stat().st_dev==target.parent.stat().st_dev and source.stat().st_size==m['source_file_size']
    original2=(ROOT/'layer2_original.q8').read_bytes();patch2=(ROOT/'layer2_edited.q8').read_bytes();restore46=(ROOT/'layer46_restored.q8').read_bytes()
    assert sha(original2)==m['original_tensor_sha256'] and sha(patch2)==m['edited_tensor_sha256'] and sha(restore46)==m['layer46_restored_sha256']
    spans=m['source_spans']
    with source.open('rb') as f:
        for layer,span in spans.items():f.seek(span['offset']);assert sha(f.read(span['size']))==span['sha256'],layer
    current_path=Path((STRATA/'fn-iq3s/boot.env').read_text().strip().split('=',1)[1]);current=read(current_path)
    assert current==read(PARENT/'strata_l44_l45_l46_20261118/candidate_config.json')
    normalized=json.loads(json.dumps(current));candidate=read(ROOT/'candidate_config.json')
    for flag in ['--native','--ple-gguf']:
        i=normalized['args'].index(flag)+1;assert normalized['args'][i]==str(source);normalized['args'][i]=str(target)
    assert normalized==candidate
    d=dict(state='waiting_for_idle',started_at=now(),from_model='44+45+46',to_model='2+44+45');write(ROOT/'deployment.json',d)
    deadline=time.monotonic()+900
    while True:
        live=get('/metrics')['live']
        if live.get('running',0)==0 and live.get('queued',0)==0:break
        if time.monotonic()>deadline:raise RuntimeError('Production requests have not drained')
        time.sleep(2)
    script=str(STRATA/'scripts/serve-fn-iq3s.sh')
    d.update(state='restarting',restart_at=now());write(ROOT/'deployment.json',d)
    subprocess.run([script,'stop'],check=True)
    try:
        with source.open('r+b') as f:
            for layer,blob in [('46',restore46),('2',patch2)]:
                span=spans[layer];f.seek(span['offset']);assert sha(f.read(span['size']))==span['sha256']
                f.seek(span['offset']);assert f.write(blob)==span['size']
            f.flush();os.fsync(f.fileno())
            after={**{k:v['sha256'] for k,v in spans.items()},'2':sha(patch2),'46':sha(restore46)}
            for layer,expected in after.items():f.seek(spans[layer]['offset']);assert sha(f.read(spans[layer]['size']))==expected
        assert source.stat().st_size==m['source_file_size']
        source.rename(target);write(Path(m['candidate_config']),candidate)
        subprocess.run([script,'start','g4x-l2l44l45-natural'],check=True)
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            try:
                health=get('/health')
                if health.get('loaded'):break
            except (OSError,ValueError):pass
            time.sleep(2)
        else:raise RuntimeError('Layer2 candidate not ready within300 seconds')
    except BaseException as error:
        subprocess.run([script,'stop'],check=False)
        active=target if target.exists() else source
        with active.open('r+b') as f:
            for layer,blob in [('2',original2),('46',restore46)]:f.seek(spans[layer]['offset']);assert f.write(blob)==len(blob)
            f.flush();os.fsync(f.fileno())
        active.rename(baseline)
        subprocess.run([script,'start','g4x-l44l45-natural'],check=True)
        d.update(state='failed_restored_44_45',error=repr(error));write(ROOT/'deployment.json',d)
        raise
    d.update(state='live',ready_at=now(),health=health);write(ROOT/'deployment.json',d)
    m.update(status='deployed_pending_panel',targeted_integrity_verified=True,layer46_restored=True,layer44_and_45_retained=True);write(ROOT/'candidate_manifest.json',m)
    for sub in ['strata_l44_l45_samedirection_20261118','strata_l44_l45_l46_20261118']:
        path=PARENT/sub/'candidate_manifest.json';prior=read(path)
        prior.update(status='preserved_results_working_file_moved',working_file_moved_to=str(target),current_working_manifest=str(ROOT/'candidate_manifest.json'))
        write(path,prior)
    print(json.dumps(d,indent=2),flush=True)

if __name__=='__main__':main()
