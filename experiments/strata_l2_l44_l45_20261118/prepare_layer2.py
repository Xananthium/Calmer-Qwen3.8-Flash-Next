#!/usr/bin/env python3
"""Prepare a complete leading-mode removal at layer 2, on the 44+45 baseline."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['STRATA_GGUF_PY']='/home/researcher/strata-122b/llama.cpp/gguf-py'
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import numpy as np

PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT=PARENT/'strata_l2_l44_l45_20261118'
PREVIOUS=PARENT/'strata_l44_l45_samedirection_20261118'
TRIPLET=PARENT/'strata_l44_l45_l46_20261118'
STRATA=Path('/home/researcher/strata-122b')
TARGET=Path('/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l2-l44-l45/Qwen3.8-Flash-Next-Gyro-M-Calmer-L2-L44-L45-TQ2_0.gguf')
CONFIG=STRATA/'fn-iq3s/cfg-g4x-l2l44l45-natural.json'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'candidate_manifest.json').exists() and not TARGET.exists() and not CONFIG.exists()
    prior=read(TRIPLET/'candidate_manifest.json');source=Path(prior['target'])
    sys.path.insert(0,str(STRATA/'src/tools'))
    from gguf_reader import GGUFFile
    from iq_pack import dequantize
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q,quants
    g=GGUFFile(source);tensors={t.name:t for t in g.tensors};t=tensors['blk.2.ffn_down_shexp.weight']
    assert t.type_name=='Q8_0' and t.shape==[640,2560]
    before={2:'61389aefa115f307db5a2f1f5436909ba40861ab6436363805fbeaaa93279602',44:prior['layer44_retained_sha256'],45:prior['layer45_retained_sha256'],46:prior['edited_tensor_sha256']}
    spans={};blobs={}
    with source.open('rb') as f:
        for layer,expected in before.items():
            tensor=tensors[f'blk.{layer}.ffn_down_shexp.weight']
            offset=g.data_start+tensor.offset;size=tensor.expected_bytes();f.seek(offset);blob=f.read(size)
            assert sha(blob)==expected,(layer,sha(blob),expected)
            spans[str(layer)]=dict(offset=offset,size=size,sha256=expected);blobs[layer]=blob
    original=blobs[2]
    w=dequantize(np.frombuffer(original,dtype=np.uint8),'Q8_0').reshape(2560,640).astype(np.float64)
    u,s,_=np.linalg.svd(w,full_matrices=False);direction=u[:,0];coeff=direction@w
    packed=quants.quantize((w-np.outer(direction,coeff)).astype(np.float32),Q.Q8_0).tobytes()
    assert len(packed)==len(original)
    actual=dequantize(np.frombuffer(packed,dtype=np.uint8),'Q8_0').reshape(w.shape).astype(np.float64)
    remaining=float(np.linalg.norm(direction@actual)/np.linalg.norm(coeff))
    assert np.isfinite(actual).all() and remaining<0.02
    restore46=(TRIPLET/'layer46_original.q8').read_bytes()
    assert sha(restore46)==prior['original_tensor_sha256'] and len(restore46)==len(blobs[46])
    for name,blob in [('layer2_original.q8',original),('layer2_edited.q8',packed),('layer46_previous.q8',blobs[46]),('layer46_restored.q8',restore46)]:
        (ROOT/name).write_bytes(blob)
    np.savez_compressed(ROOT/'layer2_edit.npz',direction=direction,coeff=coeff,singular_values=s,alpha=np.array(1.0))
    baseline_path=Path(prior['rollback']['restore_path']);baseline_config=Path(prior['rollback']['restore_config'])
    baseline=read(baseline_config);candidate=json.loads(json.dumps(baseline))
    for flag in ['--native','--ple-gguf']:
        i=candidate['args'].index(flag)+1;assert candidate['args'][i]==str(baseline_path);candidate['args'][i]=str(TARGET)
    write(ROOT/'baseline_natural_config.json',baseline);write(ROOT/'candidate_config.json',candidate)
    for name in ['protocol.json','loop_protocol.json']:
        protocol=read(TRIPLET/name)
        if name=='protocol.json':
            cases={c['case']:c for c in protocol['cases']};protocol['cases']=[cases[k] for k in ['05','01','07','08']]
        protocol.update(created_at=datetime.now(timezone.utc).isoformat(),objective='Remove the entire leading shared-expert output mode in layer2 on the layer44+45 baseline; undo layer46.',
            baseline_model=str(baseline_path),edited_model=str(TARGET),
            selection='Same four core prompts and seeds; math first. Same eight short loop/control probes. All requests frozen before candidate inference.',
            limitations=['Small exploratory panel, not a general benchmark.','Layer44+45 is the immediate baseline; stored requests are identical, with75000 test ceiling and no forced reasoning closure.',
                'Core order changed to run math first at user priority.','Shared production traffic and speculative decoding permit same-seed variation.'])
        write(ROOT/name,protocol)
    m=dict(candidate='l2-svd1-alpha1-plus-l44-d45-alpha1-plus-l45-svd1',source=str(source),target=str(TARGET),
        parent_manifest=str(PREVIOUS/'candidate_manifest.json'),physical_source_manifest=str(TRIPLET/'candidate_manifest.json'),
        applied_layers=[2,44,45],removed_edit_layers=[46],added_tensor=t.name,offset=spans['2']['offset'],tensor_bytes=len(packed),alpha=1.0,
        original_tensor_sha256=sha(original),edited_tensor_sha256=sha(packed),source_spans=spans,
        layer44_retained_sha256=before[44],layer45_retained_sha256=before[45],layer46_restored_sha256=sha(restore46),
        formula='W2_new = W2 - outer(u2, u2.T @ W2); exact leading left singular vector; alpha=1.',
        original_top_mode_energy_share=float(s[0]**2/np.sum(w*w)),relative_frobenius_change=float(np.linalg.norm(actual-w)/np.linalg.norm(w)),
        top_mode_fraction_remaining_after_q8=remaining,candidate_config=str(CONFIG),baseline_config=str(baseline_config),baseline_model=str(baseline_path),
        source_file_size=source.stat().st_size,status='prepared_not_deployed',full_file_reread=False,test_output_ceiling=75000,
        rollback=dict(tensor=str(ROOT/'layer2_original.q8'),offset=spans['2']['offset'],sha256=sha(original),restore_path=str(baseline_path),restore_config=str(baseline_config)),
        working_notes=['Patch only offline after stopping Strata.','Undo layer46 and retain exact layer44/45 bytes.','Both native and ple-gguf must refer to the active full GGUF.',
            'Check four affected/retained tensors, file size, config, live identity; no full-file copy audit.','Preserve interrupted attempts as unfinished; never count them as natural shortening.'])
    write(ROOT/'candidate_manifest.json',m)
    print(json.dumps({k:m[k] for k in ['candidate','offset','alpha','original_top_mode_energy_share','relative_frobenius_change','top_mode_fraction_remaining_after_q8','status']},indent=2),flush=True)

if __name__=='__main__':main()
