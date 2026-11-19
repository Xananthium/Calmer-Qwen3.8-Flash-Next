#!/usr/bin/env python3
"""Materialize the layer44 recurrence edit on the layer45-only parent."""
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
OLD=PARENT/'strata_l45_svd1_20261118'
PREVIOUS=PARENT/'strata_l45plus_nearby_natural_20261118'
ROOT=PARENT/'strata_l44_l45_samedirection_20261118'
STRATA=Path('/home/researcher/strata-122b')
SOURCE=Path('/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l45-svd1/Qwen3.8-Flash-Next-Gyro-M-Calmer-L45-SVD1-TQ2_0.gguf')
TARGET=Path('/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l44-l45/Qwen3.8-Flash-Next-Gyro-M-Calmer-L44-L45-TQ2_0.gguf')

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    started=time.time()
    ROOT.mkdir(exist_ok=False);TARGET.parent.mkdir(exist_ok=True)
    temporary=TARGET.with_suffix('.gguf.partial')
    cfg=STRATA/'fn-iq3s/cfg-g4x-l44l45-natural.json'
    assert not TARGET.exists() and not temporary.exists() and not cfg.exists()
    parent=read(OLD/'manifest.json');assert parent['target']==str(SOURCE)
    patch_dir=PREVIOUS/'proposed_l44_same_direction';patch=read(patch_dir/'manifest.json')
    raw=(patch_dir/'original.q8').read_bytes();packed=(patch_dir/'candidate.q8').read_bytes()
    assert sha(raw)==patch['original_sha256'] and sha(packed)==patch['candidate_sha256']
    offset,size=patch['offset'],patch['tensor_bytes']
    with SOURCE.open('rb') as f:f.seek(offset);assert f.read(size)==raw
    source_stat=SOURCE.stat()
    assert shutil.disk_usage(TARGET.parent).free>source_stat.st_size+10*1024**3
    base_cfg=read(PREVIOUS/'baseline_natural_config.json')
    write(ROOT/'baseline_natural_config.json',base_cfg)
    write(ROOT/'baseline_reference.json',dict(directory=str(PREVIOUS/'baseline'),loop_directory=str(PREVIOUS/'loop_probe/baseline'),
        model=str(SOURCE),policy='Reuse completed layer45-only baseline. No results from the layer45+46 candidate are pooled into this baseline.'))
    for filename in ['protocol.json','loop_protocol.json']:
        protocol=read(PREVIOUS/filename)
        protocol['reused_protocol_source']=str(PREVIOUS/filename)
        protocol['objective']='Layer44 removal of the original layer45 direction, retaining layer45 edit; layer46 restored to original.'
        protocol['baseline_model']=str(SOURCE);protocol['edited_model']=str(TARGET)
        protocol['limitations']=[v.replace('adding the layer46 edit','adding the layer44 edit') for v in protocol['limitations']]
        write(ROOT/filename,protocol)
    result=dict(candidate='l44-d45-alpha1-plus-l45-svd1',source=str(SOURCE),target=str(TARGET),
        parent_manifest=str(OLD/'manifest.json'),parent_sha256=parent['candidate_sha256'],
        added_tensor=patch['tensor'],offset=offset,tensor_bytes=size,alpha=1.0,formula=patch['formula'],
        direction_source=patch['direction_source'],original_tensor_sha256=patch['original_sha256'],edited_tensor_sha256=patch['candidate_sha256'],
        relative_frobenius_change=patch['actual_relative_weight_change'],
        removed_direction_amplitude_remaining_after_q8=patch['removed_direction_amplitude_remaining_after_q8'],
        layer45_retained_sha256='8974663dc1cda1ebbb63266bc82afc622f63ac3ab38d24a718dcea36539b0c38',
        layer46_restored_sha256='4a0375c3811d552220aa2b55c3c5c5f741c96738fe0259be646404918d034749',
        candidate_config=str(cfg),baseline_reference=str(ROOT/'baseline_reference.json'),
        status='preparing',test_output_ceiling=75000)
    write(ROOT/'candidate_manifest.json',result)
    src=os.open(SOURCE,os.O_RDONLY);dst=os.open(temporary,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o664)
    copied=0;last=time.monotonic()
    try:
        try:
            fcntl.ioctl(dst,0x40049409,src);copied=source_stat.st_size;method='reflink'
        except OSError as error:
            if error.errno not in [errno.EOPNOTSUPP,errno.ENOTTY,errno.EXDEV,errno.EINVAL]:raise
            method='copy_file_range'
        while copied<source_stat.st_size:
            count=os.copy_file_range(src,dst,min(1024**3,source_stat.st_size-copied));assert count>0;copied+=count
            if time.monotonic()-last>=15:
                print(json.dumps({'stage':'copying','percent':round(100*copied/source_stat.st_size,1)}),flush=True);last=time.monotonic()
        assert os.pwrite(dst,packed,offset)==size;os.fsync(dst)
    finally:os.close(src);os.close(dst)
    assert SOURCE.stat().st_mtime_ns==source_stat.st_mtime_ns
    # The user asked to keep iteration checks focused on the edit, not reread
    # a completed 92GB kernel copy. Verify layout and the three affected identities.
    assert temporary.stat().st_size==source_stat.st_size
    sys.path.insert(0,str(STRATA/'src/tools'))
    from gguf_reader import GGUFFile
    original=GGUFFile(SOURCE);edited=GGUFFile(temporary)
    assert original.metadata==edited.metadata and original.tensors==edited.tensors and original.data_start==edited.data_start
    with temporary.open('rb') as f:
        f.seek(offset);assert sha(f.read(size))==patch['candidate_sha256']
        for layer,key in [(45,'layer45_retained_sha256'),(46,'layer46_restored_sha256')]:
            tensor=next(t for t in edited.tensors if t.name==f'blk.{layer}.ffn_down_shexp.weight')
            f.seek(edited.data_start+tensor.offset);assert sha(f.read(tensor.expected_bytes()))==result[key]
    temporary.rename(TARGET)
    candidate=json.loads(json.dumps(base_cfg))
    for flag in ['--native','--ple-gguf']:
        i=candidate['args'].index(flag)+1;assert candidate['args'][i]==str(SOURCE);candidate['args'][i]=str(TARGET)
    write(cfg,candidate);write(ROOT/'candidate_config.json',candidate)
    result.update(status='prepared_not_deployed',targeted_integrity_verified=True,
        all_unchanged_bytes_verified_against_parent=False,header_and_tensor_layout_unchanged=True,
        full_file_sha256=None,copy_completed_and_fsynced=True,
        verification_scope='Completed kernel copy, file size/layout, layer44 patch, retained layer45 and restored layer46 hashes; no full-file reread.',
        preparation_seconds=time.time()-started,copy_method=method)
    write(ROOT/'candidate_manifest.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
