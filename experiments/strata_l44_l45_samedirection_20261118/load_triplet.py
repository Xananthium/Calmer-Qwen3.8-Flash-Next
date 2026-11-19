#!/usr/bin/env python3
"""Apply the saved layer46 patch to the stopped experimental 44+45 model."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
PREVIOUS=PARENT/'strata_l44_l45_samedirection_20261118'
L46=PARENT/'strata_l45plus_nearby_natural_20261118'
ROOT=PARENT/'strata_l44_l45_l46_20261118'
STRATA=Path('/home/researcher/strata-122b')
SOURCE=Path('/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l44-l45/Qwen3.8-Flash-Next-Gyro-M-Calmer-L44-L45-TQ2_0.gguf')
TARGET=Path('/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l44-l45-l46/Qwen3.8-Flash-Next-Gyro-M-Calmer-L44-L45-L46-TQ2_0.gguf')

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def get(p):
    with urlopen('http://127.0.0.1:8003'+p,timeout=10) as f:return json.load(f)

def main():
    ROOT.mkdir(exist_ok=False);TARGET.parent.mkdir(exist_ok=True)
    assert not TARGET.exists() and SOURCE.stat().st_dev==TARGET.parent.stat().st_dev
    old=read(PREVIOUS/'candidate_manifest.json');m46=read(L46/'candidate_manifest.json')
    original=(L46/'layer46_original.q8').read_bytes();patch=(L46/'layer46_edited.q8').read_bytes()
    assert sha(original)==m46['original_tensor_sha256'] and sha(patch)==m46['edited_tensor_sha256']
    sys.path.insert(0,str(STRATA/'src/tools'))
    from gguf_reader import GGUFFile
    g=GGUFFile(SOURCE);tensors={t.name:t for t in g.tensors}
    t=tensors['blk.46.ffn_down_shexp.weight'];offset=g.data_start+t.offset
    assert t.type_name=='Q8_0' and t.expected_bytes()==len(patch)==len(original)
    before={44:old['edited_tensor_sha256'],45:old['layer45_retained_sha256'],46:m46['original_tensor_sha256']}
    with SOURCE.open('rb') as f:
        for layer,expected in before.items():
            tensor=tensors[f'blk.{layer}.ffn_down_shexp.weight'];f.seek(g.data_start+tensor.offset)
            assert sha(f.read(tensor.expected_bytes()))==expected
    config_path=STRATA/'fn-iq3s/cfg-g4x-l44l45l46-natural.json';assert not config_path.exists()
    current_path=Path((STRATA/'fn-iq3s/boot.env').read_text().strip().split('=',1)[1])
    current=read(current_path);assert current==read(Path(old['candidate_config']))
    candidate=json.loads(json.dumps(current))
    for flag in ['--native','--ple-gguf']:
        i=candidate['args'].index(flag)+1;assert candidate['args'][i]==str(SOURCE);candidate['args'][i]=str(TARGET)
    write(ROOT/'baseline_natural_config.json',current)
    write(ROOT/'baseline_reference.json',dict(directory=str(PREVIOUS/'edited'),loop_directory=str(PREVIOUS/'loop_probe/edited'),model=str(SOURCE)))
    for filename in ['protocol.json','loop_protocol.json']:
        protocol=read(PREVIOUS/filename)
        if filename=='protocol.json':protocol['cases']=[c for c in protocol['cases'] if c['case'] in ['01','05','07','08']]
        protocol.pop('revision',None)
        protocol.update(objective='Add the saved layer46 edit to layer44+45.',baseline_model=str(SOURCE),edited_model=str(TARGET),
            selection='Four core cases completed by all prior variants; loop panel retains all eight probes. Frozen before triplet inference.',
            limitations=['Small exploratory comparison, not a general benchmark.','Current layer44+45 is the immediate baseline; all requests in this pair use 75000 tokens and the same production sampler.',
                         'Prior variants were stopped by user; compare only common completed cases.','Shared traffic and speculative decoding permit same-seed variation.'])
        write(ROOT/filename,protocol)
    (ROOT/'layer46_original.q8').write_bytes(original);(ROOT/'layer46_edited.q8').write_bytes(patch)
    manifest=dict(candidate='l44-d45-alpha1-plus-l45-svd1-plus-l46-svd1',source=str(SOURCE),target=str(TARGET),
        parent_manifest=str(PREVIOUS/'candidate_manifest.json'),added_tensor=t.name,offset=offset,tensor_bytes=len(patch),
        original_tensor_sha256=sha(original),edited_tensor_sha256=sha(patch),
        layer44_retained_sha256=before[44],layer45_retained_sha256=before[45],applied_layers=[44,45,46],
        candidate_config=str(config_path),materialization='Patch only layer46 in the stopped experimental working file, then rename it. Original layer46 bytes retained for undo.',
        rollback=dict(tensor=str(ROOT/'layer46_original.q8'),sha256=sha(original),offset=offset,restore_path=str(SOURCE),restore_config=str(current_path)),
        status='prepared_patch_pending_reload',full_file_reread=False,test_output_ceiling=75000)
    write(ROOT/'candidate_manifest.json',manifest)
    deadline=time.monotonic()+900
    while True:
        live=get('/metrics')['live']
        if live.get('running',0)==0 and live.get('queued',0)==0:break
        if time.monotonic()>deadline:raise RuntimeError('User requests have not drained')
        time.sleep(2)
    d=dict(started_at=now(),state='restarting',from_model='Layer44+45',to_model='Layer44+45+46');write(ROOT/'deployment.json',d)
    script=str(STRATA/'scripts/serve-fn-iq3s.sh')
    subprocess.run([script,'stop'],check=True)
    try:
        with SOURCE.open('r+b') as f:
            f.seek(offset);assert f.read(len(original))==original
            f.seek(offset);assert f.write(patch)==len(patch);f.flush();os.fsync(f.fileno())
            for layer,expected in {**before,46:sha(patch)}.items():
                tensor=tensors[f'blk.{layer}.ffn_down_shexp.weight'];f.seek(g.data_start+tensor.offset)
                assert sha(f.read(tensor.expected_bytes()))==expected
        SOURCE.rename(TARGET);write(config_path,candidate);write(ROOT/'candidate_config.json',candidate)
        subprocess.run([script,'start','g4x-l44l45l46-natural'],check=True)
    except BaseException:
        subprocess.run([script,'stop'],check=False)
        active=TARGET if TARGET.exists() else SOURCE
        with active.open('r+b') as f:f.seek(offset);f.write(original);f.flush();os.fsync(f.fileno())
        if active==TARGET:TARGET.rename(SOURCE)
        subprocess.run([script,'start','g4x-l44l45-natural'],check=True)
        raise
    old.update(status='preserved_results_model_restorable_via_layer46_undo',working_file_moved_to=str(TARGET),rollback_manifest=str(ROOT/'candidate_manifest.json'));write(PREVIOUS/'candidate_manifest.json',old)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        try:
            health=get('/health')
            if health.get('loaded'):
                d.update(state='live',ready_at=now(),health=health);write(ROOT/'deployment.json',d)
                manifest.update(status='deployed_pending_panel',targeted_integrity_verified=True,layer44_and_45_retained=True)
                write(ROOT/'candidate_manifest.json',manifest)
                print(json.dumps(d,indent=2),flush=True);return
        except (OSError,ValueError):pass
        time.sleep(2)
    raise RuntimeError('Triplet is not ready within 300 seconds')

if __name__=='__main__':main()
