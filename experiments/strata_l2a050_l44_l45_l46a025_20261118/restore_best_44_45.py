#!/usr/bin/env python3
"""Restore the selected44+45 model and make its preset the persistent boot target."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
from pathlib import Path
from urllib.request import urlopen

PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT=PARENT/'strata_l2a050_l44_l45_l46a025_20261118'
BEST=PARENT/'strata_l44_l45_samedirection_20261118'
STRATA=Path('/home/researcher/strata-122b')
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def get(p):
    with urlopen('http://127.0.0.1:8003'+p,timeout=10) as response:return json.load(response)

def main():
    m=read(ROOT/'candidate_manifest.json');source=Path(m['target']);restore=m['rollback_44_45'];target=Path(restore['model'])
    assert source.exists() and not target.exists()
    target.parent.mkdir(exist_ok=True);spans=m['source_spans'];before=m['expected_tensor_hashes']
    patches={layer:Path(path).read_bytes() for layer,path in restore['tensors'].items()}
    expected={**before,**{layer:m['edits'][layer]['original_tensor_sha256'] for layer in patches}}
    for layer,blob in patches.items():assert sha(blob)==expected[layer] and len(blob)==spans[layer]['size']
    with source.open('rb') as f:
        for layer,digest in before.items():f.seek(spans[layer]['offset']);assert sha(f.read(spans[layer]['size']))==digest
    cfg=read(Path(restore['config']));current=read(Path((STRATA/'fn-iq3s/boot.env').read_text().strip().split('=',1)[1]))
    assert current==read(ROOT/'candidate_config.json')
    normalized=json.loads(json.dumps(current))
    for flag in ['--native','--ple-gguf']:normalized['args'][normalized['args'].index(flag)+1]=str(target)
    assert normalized==cfg
    record=dict(state='restoring',started_at=now(),model=str(target),config=restore['config'],expected_tensor_hashes=expected)
    write(ROOT/'restore_best.json',record)
    script=str(STRATA/'scripts/serve-fn-iq3s.sh');subprocess.run([script,'stop'],check=True)
    try:
        with source.open('r+b') as f:
            for layer,blob in patches.items():
                f.seek(spans[layer]['offset']);assert sha(f.read(spans[layer]['size']))==before[layer]
                f.seek(spans[layer]['offset']);assert f.write(blob)==len(blob)
            f.flush();os.fsync(f.fileno())
            for layer,digest in expected.items():f.seek(spans[layer]['offset']);assert sha(f.read(spans[layer]['size']))==digest
        assert source.stat().st_size==m['source_file_size'];source.rename(target)
        subprocess.run([script,'start','g4x-l44l45-natural'],check=True)
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            try:
                health=get('/health')
                if health.get('loaded'):break
            except (OSError,ValueError):pass
            time.sleep(2)
        else:raise RuntimeError('Restored model did not become ready')
    except BaseException as error:
        subprocess.run([script,'stop'],check=False)
        subprocess.run([script,'start','g4x-unedited-natural'],check=True)
        record.update(state='restore_failed_original_baseline_running',error=repr(error));write(ROOT/'restore_best.json',record)
        raise
    boot=(STRATA/'fn-iq3s/boot.env').read_text().strip();assert boot=='STRATA_CFG='+restore['config']
    subprocess.run(['systemctl','--user','enable','strata-fn-serve.service'],check=True)
    enabled=subprocess.check_output(['systemctl','--user','is-enabled','strata-fn-serve.service'],text=True).strip();assert enabled=='enabled'
    router=subprocess.check_output(['systemctl','is-enabled','qwen-reasoning-router.service'],text=True).strip();assert router=='enabled'
    linger=subprocess.check_output(['loginctl','show-user','isnotreal','-p','Linger'],text=True).strip();assert linger=='Linger=yes'
    processes=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:args=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        except (OSError,UnicodeDecodeError):continue
        if args and args[0]==str(STRATA/'build/strata') and '--serve' in args:
            for flag in ['--native','--ple-gguf']:assert args[args.index(flag)+1]==str(target)
            processes.append(int(p.name))
    assert len(processes)==1
    record.update(state='restored_live_and_reboot_default',ready_at=now(),health=health,boot_env=boot,service_enabled=enabled,
        router_enabled=router,user_linger=linger,engine_pids=processes,reboot_performed=False)
    write(ROOT/'restore_best.json',record)
    m.update(status='experiment_stopped_best44_45_restored',working_file_moved_to=str(target),restoration_record=str(ROOT/'restore_best.json'));write(ROOT/'candidate_manifest.json',m)
    best=read(BEST/'candidate_manifest.json');best.update(status='selected_production_and_reboot_default',target=str(target),
        working_file_moved_to=None,current_working_manifest=str(BEST/'candidate_manifest.json'),restoration_record=str(ROOT/'restore_best.json'));write(BEST/'candidate_manifest.json',best)
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
