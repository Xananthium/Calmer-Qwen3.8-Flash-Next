#!/usr/bin/env python3
"""Apply half of the original layer2 mode removal and one quarter of layer46."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['STRATA_GGUF_PY']='/home/researcher/strata-122b/llama.cpp/gguf-py'
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import numpy as np

PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT=PARENT/'strata_l2a050_l44_l45_l46a025_20261118'
L2=PARENT/'strata_l2_l44_l45_20261118'
L46=PARENT/'strata_l45plus_nearby_natural_20261118'
UNEDITED=PARENT/'strata_unedited_natural_20261118'
STRATA=Path('/home/researcher/strata-122b')
TARGET=Path('/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l2a050-l44-l45-l46a025/Qwen3.8-Flash-Next-Gyro-M-Calmer-L2A050-L44-L45-L46A025-TQ2_0.gguf')
CONFIG=STRATA/'fn-iq3s/cfg-g4x-l2a050-l44l45-l46a025-natural.json'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()

def ensure_offline(source):
    cfg=Path((STRATA/'fn-iq3s/boot.env').read_text().strip().split('=',1)[1])
    assert read(cfg)==read(UNEDITED/'candidate_config.json')
    for proc in Path('/proc').glob('[0-9]*'):
        try:args=proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        except (OSError,UnicodeDecodeError):continue
        assert str(source) not in args,('Experimental working file is in use',proc.name)

def main():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'candidate_manifest.json').exists() and not CONFIG.exists()
    old=read(L2/'candidate_manifest.json');source=Path(old['target']);assert source.exists() and not TARGET.exists()
    ensure_offline(source);TARGET.parent.mkdir(exist_ok=True)
    assert source.stat().st_dev==TARGET.parent.stat().st_dev
    sys.path.insert(0,str(STRATA/'src/tools'))
    from gguf_reader import GGUFFile
    from iq_pack import dequantize
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q,quants
    g=GGUFFile(source);spans={};before_blobs={}
    before={'2':old['edited_tensor_sha256'],'44':old['layer44_retained_sha256'],'45':old['layer45_retained_sha256'],'46':old['layer46_restored_sha256']}
    with source.open('rb') as f:
        for layer,expected in before.items():
            t=next(t for t in g.tensors if t.name==f'blk.{layer}.ffn_down_shexp.weight');assert t.type_name=='Q8_0' and t.shape==[640,2560]
            offset=g.data_start+t.offset;size=t.expected_bytes();f.seek(offset);blob=f.read(size);assert sha(blob)==expected
            spans[layer]=dict(offset=offset,size=size,sha256=expected);before_blobs[layer]=blob
    edits={};patches={};expected=dict(before)
    for layer,alpha,directory,original_hash in [('2',0.5,L2,old['original_tensor_sha256']),('46',0.25,L46,old['layer46_restored_sha256'])]:
        original=(directory/f'layer{layer}_original.q8').read_bytes();assert sha(original)==original_hash
        saved=np.load(directory/f'layer{layer}_edit.npz');direction=saved['direction'];coeff=saved['coeff']
        w=dequantize(np.frombuffer(original,dtype=np.uint8),'Q8_0').reshape(2560,640).astype(np.float64)
        assert abs(np.linalg.norm(direction)-1)<1e-10 and np.allclose(direction@w,coeff,rtol=1e-10,atol=1e-12)
        packed=quants.quantize((w-alpha*np.outer(direction,coeff)).astype(np.float32),Q.Q8_0).tobytes()
        assert len(packed)==spans[layer]['size']
        actual=dequantize(np.frombuffer(packed,dtype=np.uint8),'Q8_0').reshape(w.shape).astype(np.float64)
        remaining=float(np.linalg.norm(direction@actual)/np.linalg.norm(coeff))
        assert np.isfinite(actual).all() and abs(remaining-(1-alpha))<0.02
        for suffix,blob in [('original',original),('previous',before_blobs[layer]),('edited',packed)]:
            (ROOT/f'layer{layer}_{suffix}.q8').write_bytes(blob)
        np.savez_compressed(ROOT/f'layer{layer}_edit.npz',direction=direction,coeff=coeff,alpha=np.array(alpha))
        patches[layer]=packed;expected[layer]=sha(packed)
        edits[layer]=dict(alpha=alpha,original_tensor_sha256=original_hash,edited_tensor_sha256=sha(packed),
            direction_source=str(directory/f'layer{layer}_edit.npz'),original_top_mode_energy_share=float(np.sum(coeff*coeff)/np.sum(w*w)),
            relative_frobenius_change=float(np.linalg.norm(actual-w)/np.linalg.norm(w)),top_mode_fraction_remaining_after_q8=remaining)
    cfg=read(UNEDITED/'candidate_config.json');candidate=json.loads(json.dumps(cfg))
    for flag in ['--native','--ple-gguf']:candidate['args'][candidate['args'].index(flag)+1]=str(TARGET)
    write(ROOT/'candidate_config.json',candidate);write(CONFIG,candidate)
    for name in ['protocol.json','loop_protocol.json']:
        protocol=read(UNEDITED/name)
        protocol.update(created_at=datetime.now(timezone.utc).isoformat(),objective='Retain44+45 and add half-strength layer2 plus quarter-strength layer46 suppression using the original directions.',
            baseline_model=old['baseline_model'],edited_model=str(TARGET),
            selection='Same core requests, math first, and same eight short probes as the completed unedited baseline.',
            limitations=['Small exploratory panel, not a general benchmark.','Same production settings and runtime control vector; no forced reasoning closure;75000-token per-request ceiling.',
                'Changing both partial strengths together tests their combination; it does not isolate either contribution.',
                'Shared production traffic and speculative decoding permit same-seed variation.'])
        write(ROOT/name,protocol)
    m=dict(candidate='l44-l45-plus-l2-alpha0.50-l46-alpha0.25',source=str(source),target=str(TARGET),applied_layers=[2,44,45,46],
        edits=edits,expected_tensor_hashes=expected,source_spans=spans,source_file_size=source.stat().st_size,
        candidate_config=str(CONFIG),baseline_44_45=str(PARENT/'strata_l44_l45_samedirection_20261118/edited'),
        baseline_unedited=str(UNEDITED/'baseline'),status='prepared_patch_pending',full_file_reread=False,test_output_ceiling=75000,
        formula='W_layer_new = W_layer_original - alpha * outer(saved_original_direction, saved_original_coeff).',
        rollback_previous=dict(model=str(source),config=old['candidate_config'],tensors={layer:str(ROOT/f'layer{layer}_previous.q8') for layer in ['2','46']}),
        rollback_44_45=dict(model=old['baseline_model'],config=old['baseline_config'],tensors={layer:str(ROOT/f'layer{layer}_original.q8') for layer in ['2','46']}),
        working_notes=['Original model remains live while the separate experimental file is patched offline.','Retain exact layer44/45 bytes; rebuild2/46 from saved original weights and directions.',
            'Only targeted tensor checks and file-size checks; no full-file reread.','Native and ple-gguf references must match the same full model.'])
    write(ROOT/'candidate_manifest.json',m);ensure_offline(source)
    try:
        with source.open('r+b') as f:
            for layer,blob in patches.items():
                s=spans[layer];f.seek(s['offset']);assert sha(f.read(s['size']))==before[layer]
                f.seek(s['offset']);assert f.write(blob)==s['size']
            f.flush();os.fsync(f.fileno())
            for layer,digest in expected.items():f.seek(spans[layer]['offset']);assert sha(f.read(spans[layer]['size']))==digest
        assert source.stat().st_size==m['source_file_size'];source.rename(TARGET)
    except BaseException:
        active=TARGET if TARGET.exists() else source
        with active.open('r+b') as f:
            for layer in ['2','46']:f.seek(spans[layer]['offset']);f.write(before_blobs[layer])
            f.flush();os.fsync(f.fileno())
        if active==TARGET:TARGET.rename(source)
        raise
    m.update(status='prepared_not_deployed',targeted_integrity_verified=True);write(ROOT/'candidate_manifest.json',m)
    old.update(status='preserved_results_working_file_moved',working_file_moved_to=str(TARGET),current_working_manifest=str(ROOT/'candidate_manifest.json'));write(L2/'candidate_manifest.json',old)
    print(json.dumps(dict(status=m['status'],edits=edits),indent=2),flush=True)

if __name__=='__main__':main()
