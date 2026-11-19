#!/usr/bin/env python3
"""Prepare a small layer44 patch for the mapped layer45 direction; do not deploy."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['STRATA_GGUF_PY']='/home/researcher/strata-122b/llama.cpp/gguf-py'
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import natural_panel as n

def main():
    sys.path.insert(0,'/home/researcher/strata-122b/src/tools')
    from gguf_reader import GGUFFile
    from iq_pack import dequantize
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q,quants
    out=n.ROOT/'proposed_l44_same_direction';out.mkdir(exist_ok=False)
    g=GGUFFile(n.TARGET);t=next(t for t in g.tensors if t.name=='blk.44.ffn_down_shexp.weight')
    assert t.type_name=='Q8_0' and t.shape==[640,2560]
    with open(n.TARGET,'rb') as f:f.seek(g.data_start+t.offset);raw=f.read(t.expected_bytes())
    sha=lambda b:hashlib.sha256(b).hexdigest()
    assert sha(raw)=='eca71b05df1c2d66e55411e4420e09d4d6b9e96b80c04c012d0af0afcdedb1bf'
    w=dequantize(np.frombuffer(raw,dtype=np.uint8),'Q8_0').reshape(2560,640).astype(np.float64)
    d=np.load(n.OLD/'edit.npz')['direction'];d=d/np.linalg.norm(d)
    coeff=d@w;ideal=w-np.outer(d,coeff)
    packed=quants.quantize(ideal.astype(np.float32),Q.Q8_0).tobytes()
    q=dequantize(np.frombuffer(packed,dtype=np.uint8),'Q8_0').reshape(w.shape).astype(np.float64)
    remaining=float(np.linalg.norm(d@q)/np.linalg.norm(coeff))
    assert len(packed)==len(raw) and np.isfinite(q).all() and remaining<0.02
    (out/'original.q8').write_bytes(raw);(out/'candidate.q8').write_bytes(packed)
    np.savez_compressed(out/'edit.npz',direction=d,coeff=coeff,alpha=np.array(1.0))
    result=dict(at=datetime.now(timezone.utc).isoformat(),state='prepared_patch_only_not_applied',
        parent_model=n.TARGET,parent_manifest=str(n.ROOT/'candidate_manifest.json'),
        tensor=t.name,offset=g.data_start+t.offset,tensor_bytes=len(raw),alpha=1.0,
        formula='W44_new = W44 - outer(d45, d45.T @ W44)',
        direction_source=str(n.OLD/'edit.npz'),
        reason='Remove the mapped recurrence of the original layer45 direction from layer44; it is spread across multiple leading layer44 modes.',
        original_sha256=sha(raw),candidate_sha256=sha(packed),
        removed_direction_energy_share=float(np.sum(coeff*coeff)/np.sum(w*w)),
        actual_relative_weight_change=float(np.linalg.norm(q-w)/np.linalg.norm(w)),
        removed_direction_amplitude_remaining_after_q8=remaining,
        model_files_modified=False,production_restarted=False,
        evaluation_status='Not tested. Current production remains the 45+46 candidate during its matched comparison.')
    n.harness.write(out/'manifest.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
