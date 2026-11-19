#!/usr/bin/env python3
"""Load original weights under the unchanged natural-stop production settings."""
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen

PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT=PARENT/'strata_unedited_natural_20261118'
PREVIOUS=PARENT/'strata_l2_l44_l45_20261118'
STRATA=Path('/home/researcher/strata-122b')
CONFIG=STRATA/'fn-iq3s/cfg-g4x-unedited-natural.json'
MODEL='/home/researcher/models/qwen38-flash-next-gyro-m/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def get(p):
    with urlopen('http://127.0.0.1:8003'+p,timeout=10) as response:return json.load(response)

def main():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'candidate_manifest.json').exists()
    previous_config=Path((STRATA/'fn-iq3s/boot.env').read_text().strip().split('=',1)[1])
    current=read(previous_config);assert current==read(PREVIOUS/'candidate_config.json')
    candidate=json.loads(json.dumps(current))
    for flag in ['--native','--ple-gguf']:candidate['args'][candidate['args'].index(flag)+1]=MODEL
    if CONFIG.exists():assert read(CONFIG)==candidate
    write(CONFIG,candidate);write(ROOT/'candidate_config.json',candidate);write(ROOT/'previous_config.json',current)
    for key,value in [('answer_room',False),('reasoning_budget_tokens',0),('reasoning_close_retry',False),('reasoning_loop_recovery',False),('repeat_stop_tokens',0)]:assert candidate.get(key)==value
    hashes={'2':read(PREVIOUS/'candidate_manifest.json')['original_tensor_sha256'],
        '44':read(PARENT/'strata_l44_l45_samedirection_20261118/candidate_manifest.json')['original_tensor_sha256'],
        '45':read(PARENT/'strata_l45_svd1_20261118/manifest.json')['original_tensor_sha256'],
        '46':read(PREVIOUS/'candidate_manifest.json')['layer46_restored_sha256']}
    sys.path.insert(0,str(STRATA/'src/tools'))
    from gguf_reader import GGUFFile
    g=GGUFFile(MODEL)
    with open(MODEL,'rb') as f:
        for layer,expected in hashes.items():
            t=next(t for t in g.tensors if t.name==f'blk.{layer}.ffn_down_shexp.weight');f.seek(g.data_start+t.offset)
            assert hashlib.sha256(f.read(t.expected_bytes())).hexdigest()==expected,layer
    for name in ['protocol.json','loop_protocol.json']:
        protocol=read(PREVIOUS/name)
        protocol.update(created_at=now(),objective='Establish the missing original-weight baseline under the same natural-stop production settings.',
            baseline_model=MODEL,edited_model=None,
            selection='Same four core prompts/seeds, math first; same eight short loop/control prompts. Requests unchanged from the layer2+44+45 experiment.',
            limitations=['Small exploratory panel, not a general benchmark.','Unedited means original GGUF weights. Existing production control vector and all runtime settings are retained.',
                'Earlier original-weight tests used8192-token requests and server-inserted wrap-ups; they are not this baseline.',
                'Shared production traffic and speculative decoding permit same-seed variation.'])
        write(ROOT/name,protocol)
    m=dict(candidate='original-weights-natural-stop-production-baseline',target=MODEL,source=MODEL,applied_layers=[],
        expected_tensor_hashes=hashes,candidate_config=str(CONFIG),previous_live_config=str(previous_config),
        previous_live_model=current['args'][current['args'].index('--native')+1],model_size=Path(MODEL).stat().st_size,
        status='prepared_not_deployed',test_output_ceiling=75000,full_file_reread=False,
        runtime_control_vector_retained=True,weight_bytes_modified=False,leave_running_after_tests=True)
    write(ROOT/'candidate_manifest.json',m)
    write(ROOT/'earlier_baseline_audit.json',dict(directory=str(PARENT/'strata_l45_svd1_20261118/baseline'),
        request_max_tokens=8192,forced_wrap_marker_observed_in_cases=['01','05'],comparable_to_natural_stop_panel=False))
    d=dict(state='waiting_for_idle',started_at=now(),to_model='original unedited weights');write(ROOT/'deployment.json',d)
    deadline=time.monotonic()+900
    while True:
        live=get('/metrics')['live']
        if live.get('running',0)==0 and live.get('queued',0)==0:break
        if time.monotonic()>deadline:raise RuntimeError('Production requests have not drained')
        time.sleep(2)
    script=str(STRATA/'scripts/serve-fn-iq3s.sh');d.update(state='restarting',restart_at=now());write(ROOT/'deployment.json',d)
    subprocess.run([script,'stop'],check=True)
    try:
        subprocess.run([script,'start','g4x-unedited-natural'],check=True)
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            try:
                health=get('/health')
                if health.get('loaded'):break
            except (OSError,ValueError):pass
            time.sleep(2)
        else:raise RuntimeError('Original model failed to become ready within300 seconds')
    except BaseException as error:
        subprocess.run([script,'stop'],check=False)
        subprocess.run([script,'start','g4x-l2l44l45-natural'],check=True)
        d.update(state='failed_restored_previous_model',error=repr(error));write(ROOT/'deployment.json',d)
        raise
    d.update(state='live',ready_at=now(),health=health);write(ROOT/'deployment.json',d)
    m.update(status='deployed_pending_panel',targeted_integrity_verified=True);write(ROOT/'candidate_manifest.json',m)
    print(json.dumps(d,indent=2),flush=True)

if __name__=='__main__':main()
