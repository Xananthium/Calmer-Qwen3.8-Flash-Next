#!/usr/bin/env python3
"""Add one layer-46 deflation to the immutable layer-45 model copy."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['STRATA_GGUF_PY']='/home/researcher/strata-122b/llama.cpp/gguf-py'
import errno
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

STRATA=Path('/home/researcher/strata-122b')
OLD=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45_svd1_20261118')
ROOT=OLD.parent/'strata_l45plus_nearby_natural_20261118'
SOURCE=Path('/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l45-svd1/Qwen3.8-Flash-Next-Gyro-M-Calmer-L45-SVD1-TQ2_0.gguf')
TARGET=Path('/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l45-l46-svd1/Qwen3.8-Flash-Next-Gyro-M-Calmer-L45-L46-SVD1-TQ2_0.gguf')

def sha(b):
    return hashlib.sha256(b).hexdigest()

def main():
    started=time.time()
    sys.path.insert(0,str(STRATA/'src/tools'))
    from gguf_reader import GGUFFile
    from iq_pack import dequantize
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q,quants
    parent=json.loads((OLD/'manifest.json').read_text())
    assert parent['target']==str(SOURCE)
    expected_parent=parent['candidate_sha256']
    TARGET.parent.mkdir(exist_ok=True)
    temporary=TARGET.with_suffix('.gguf.partial')
    config=STRATA/'fn-iq3s/cfg-g4x-l45l46svd1-natural.json'
    assert not TARGET.exists() and not temporary.exists() and not config.exists()
    source_stat=SOURCE.stat()
    assert shutil.disk_usage(TARGET.parent).free>source_stat.st_size+10*1024**3
    g=GGUFFile(SOURCE)
    t=next(t for t in g.tensors if t.name=='blk.46.ffn_down_shexp.weight')
    assert t.type_name=='Q8_0' and t.shape==[640,2560]
    offset,size=g.data_start+t.offset,t.expected_bytes()
    with SOURCE.open('rb') as f:
        f.seek(offset);original=f.read(size)
    assert sha(original)=='4a0375c3811d552220aa2b55c3c5c5f741c96738fe0259be646404918d034749'
    w=dequantize(np.frombuffer(original,dtype=np.uint8),'Q8_0').reshape(2560,640).astype(np.float64)
    u,s,_=np.linalg.svd(w,full_matrices=False)
    direction=u[:,0];coeff=direction@w
    ideal=w-np.outer(direction,coeff)
    packed=quants.quantize(ideal.astype(np.float32),Q.Q8_0).tobytes()
    assert len(packed)==size
    actual=dequantize(np.frombuffer(packed,dtype=np.uint8),'Q8_0').reshape(w.shape).astype(np.float64)
    remaining=np.linalg.norm(direction@actual)/np.linalg.norm(coeff)
    assert np.isfinite(actual).all() and remaining<0.02
    np.savez_compressed(ROOT/'layer46_edit.npz',direction=direction,coeff=coeff,singular_values=s,alpha=np.array(1.0))
    (ROOT/'layer46_original.q8').write_bytes(original)
    (ROOT/'layer46_edited.q8').write_bytes(packed)
    result=dict(candidate='l45-svd1-plus-l46-svd1-alpha1',parent_manifest=str(OLD/'manifest.json'),
                source=str(SOURCE),target=str(TARGET),parent_sha256=expected_parent,
                added_tensor=t.name,offset=offset,tensor_bytes=size,alpha=1.0,
                original_tensor_sha256=sha(original),edited_tensor_sha256=sha(packed),
                formula='W46_new = W46 - outer(u46, u46.T @ W46); u46 is exact leading left singular vector',
                original_top_mode_energy_share=float(s[0]**2/np.sum(w*w)),
                relative_frobenius_change=float(np.linalg.norm(actual-w)/np.linalg.norm(w)),
                top_mode_fraction_remaining_after_q8=float(remaining),
                layer45_retained_sha256=parent['edited_tensor_sha256'],
                baseline_condition='Existing layer45 edit, forced endings disabled for both arms',
                baseline_config_name='g4x-l45svd1-natural',candidate_config=str(config),
                background_preparation_during_baseline=True,
                timing_limit='Copying may affect shared production latency; completion tokens and natural correctness are primary.')
    print(json.dumps({'stage':'tensor_prepared',**result}),flush=True)
    # Kernel copy first, then patch. Never hard-link the live model.
    src=os.open(SOURCE,os.O_RDONLY)
    dst=os.open(temporary,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o664)
    copied=0;last=time.monotonic()
    try:
        try:
            fcntl.ioctl(dst,0x40049409,src) # FICLONE; fall back when unsupported.
            copied=source_stat.st_size
            method='reflink'
        except OSError as error:
            if error.errno not in [errno.EOPNOTSUPP,errno.ENOTTY,errno.EXDEV,errno.EINVAL]:raise
            method='copy_file_range'
        while copied<source_stat.st_size:
            n=os.copy_file_range(src,dst,min(1024**3,source_stat.st_size-copied))
            assert n>0
            copied+=n
            if time.monotonic()-last>=15:
                print(json.dumps({'stage':'copying','percent':round(copied/source_stat.st_size*100,1)}),flush=True)
                last=time.monotonic()
        assert os.pwrite(dst,packed,offset)==size
        os.fsync(dst)
    finally:
        os.close(src);os.close(dst)
    assert SOURCE.stat().st_mtime_ns==source_stat.st_mtime_ns
    # One readback verifies every byte against the known parent's SHA. Before
    # the changed span the hashes are identical; fork SHA state only at the edit.
    prefix=hashlib.sha256();parent_hash=candidate_hash=None
    position=0;last=time.monotonic()
    print(json.dumps({'stage':'verifying_full_candidate','copy_method':method}),flush=True)
    with temporary.open('rb') as f:
        while chunk:=f.read(32*1024*1024):
            end=position+len(chunk)
            lo,hi=max(position,offset),min(end,offset+size)
            if lo<hi:
                if parent_hash is None:
                    parent_hash=prefix.copy();candidate_hash=prefix.copy()
                assert chunk[lo-position:hi-position]==packed[lo-offset:hi-offset]
                restored=chunk[:lo-position]+original[lo-offset:hi-offset]+chunk[hi-position:]
                parent_hash.update(restored);candidate_hash.update(chunk)
            elif parent_hash is None:
                prefix.update(chunk)
            else:
                parent_hash.update(chunk);candidate_hash.update(chunk)
            position=end
            if time.monotonic()-last>=15:
                print(json.dumps({'stage':'verifying','percent':round(position/source_stat.st_size*100,1)}),flush=True)
                last=time.monotonic()
    assert position==source_stat.st_size and parent_hash.hexdigest()==expected_parent
    g2=GGUFFile(temporary)
    assert g.metadata==g2.metadata and g.tensors==g2.tensors and g.data_start==g2.data_start
    t45=next(t for t in g2.tensors if t.name=='blk.45.ffn_down_shexp.weight')
    with temporary.open('rb') as f:
        f.seek(g2.data_start+t45.offset)
        assert sha(f.read(t45.expected_bytes()))==parent['edited_tensor_sha256']
    temporary.rename(TARGET)
    baseline=json.loads((ROOT/'baseline_natural_config.json').read_text())
    candidate=json.loads(json.dumps(baseline))
    for flag in ['--native','--ple-gguf']:
        i=candidate['args'].index(flag)+1
        assert candidate['args'][i]==str(SOURCE)
        candidate['args'][i]=str(TARGET)
    assert candidate['answer_room'] is False and candidate['reasoning_budget_tokens']==0
    config.write_text(json.dumps(candidate,indent=2)+'\n')
    (ROOT/'candidate_config.json').write_text(json.dumps(candidate,indent=2)+'\n')
    result.update(candidate_sha256=candidate_hash.hexdigest(),all_unchanged_bytes_verified_against_parent=True,
                  header_and_tensor_layout_unchanged=True,preparation_seconds=time.time()-started,
                  copy_method=method,status='prepared_not_deployed',changed_model_references=['--native','--ple-gguf'])
    (ROOT/'candidate_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
