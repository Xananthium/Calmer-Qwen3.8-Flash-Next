#!/usr/bin/env python3
"""Static, read-only map of candidate output directions across all shared experts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['STRATA_GGUF_PY']='/home/researcher/strata-122b/llama.cpp/gguf-py'
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import natural_panel as n

ROOT=n.ROOT
SCAN=n.OLD.parent/'strata_readonly_20261118'
STRATA=Path('/home/researcher/strata-122b')

def main():
    started=time.monotonic()
    sys.path.insert(0,str(STRATA/'src/tools'))
    from gguf_reader import GGUFFile
    from iq_pack import dequantize
    model=Path(n.SOURCE);before=model.stat()
    g=GGUFFile(model);tensors={t.name:t for t in g.tensors}
    references={'removed_l45':np.load(n.OLD/'edit.npz')['direction'],
                'candidate_l46':np.load(ROOT/'layer46_edit.npz')['direction']}
    original_scan=json.loads((SCAN/'inspection.json').read_text())
    cv=GGUFFile(original_scan['control_vectors']['path'])
    t=cv.tensors[0]
    with open(cv.path if hasattr(cv,'path') else original_scan['control_vectors']['path'],'rb') as f:
        f.seek(cv.data_start+t.offset);references['control_vector']=np.frombuffer(f.read(t.expected_bytes()),dtype='<f4').astype(np.float64)
    for k,v in references.items():references[k]=v/np.linalg.norm(v)
    rows=[];directions={};bytes_read=0
    with model.open('rb') as f:
        def read(name):
            nonlocal bytes_read
            t=tensors[name];f.seek(g.data_start+t.offset);raw=f.read(t.expected_bytes());bytes_read+=len(raw)
            assert len(raw)==t.expected_bytes()
            w=dequantize(np.frombuffer(raw,dtype=np.uint8),t.type_name).reshape(tuple(reversed(t.shape))).astype(np.float64)
            assert np.isfinite(w).all()
            return w,hashlib.sha256(raw).hexdigest()
        for layer in range(48):
            name=f'blk.{layer}.ffn_down_shexp.weight';w,digest=read(name)
            u,s,_=np.linalg.svd(w,full_matrices=False);energy=float(np.sum(w*w))
            directions[str(layer)]=u[:,:4]
            shared={k:dict(top_direction_absolute_cosine=float(abs(u[:,0]@d)),
                          top4_subspace_squared_overlap=float(np.sum((u[:,:4].T@d)**2)),
                          matrix_energy_along_direction=float(np.sum((d@w)**2)/energy)) for k,d in references.items()}
            router,rhash=read(f'blk.{layer}.ffn_gate_inp.weight');re=float(np.sum(router*router))
            rn=np.linalg.norm(router,axis=1)
            rows.append(dict(layer=layer,tensor=name,encoded_sha256=digest,
                top4_energy_shares=(s[:4]**2/energy).tolist(),rank1_removal_relative_norm=float(s[0]/np.sqrt(energy)),
                reference_directions=shared,router=dict(encoded_sha256=rhash,row_norm_max_over_median=float(rn.max()/np.median(rn)),
                input_direction_energy_shares={k:float(np.sum((router@d)**2)/re) for k,d in references.items()}),
                existing_control_projection_enabled=4<=layer<=44,already_edited=layer==45))
            if layer%8==7:print(json.dumps({'shared_layers_mapped':layer+1,'seconds':round(time.monotonic()-started,2)}),flush=True)
    sampled=[]
    saved=np.load(SCAN/'spectral_hypotheses.npz')
    for row in original_scan['rows']:
        if row['role']!='routed_down_sample':continue
        v=saved[f'layer_{row["layer"]}_routed_down_sample_{row["expert"]}'].astype(np.float64)
        sampled.append(dict(layer=row['layer'],expert=row['expert'],tensor=row['tensor'],
            encoded_sha256=row['encoded_sha256'],approx_top_energy=row['approx_principal_energy_shares'][0],
            reference_directions={k:dict(top_direction_absolute_cosine=float(abs(v[:,0]@d)),
            top4_subspace_squared_overlap=float(np.sum((v.T@d)**2))) for k,d in references.items()}))
    after=model.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),model=str(model),baseline='Existing layer45 edit',
        coverage=dict(shared_matrices_exact_svd=48,routers=48,routed_experts_sampled=192,total_routed_experts=48*512,
                      sampled_expert_ids=[0,127,255,511],live_activations=False,attention_and_state_space_weights=False),
        method='Exact CPU SVD and reference-direction projections for all current shared output matrices; router input projection; saved rank4 approximations for the four routed expert samples per layer.',
        reference_direction_sources={'removed_l45':str(n.OLD/'edit.npz'),'candidate_l46':str(ROOT/'layer46_edit.npz'),'control_vector':original_scan['control_vectors']['path']},
        rows=rows,routed_samples=sampled,
        rankings={k:dict(by_top_direction_alignment=[r['layer'] for r in sorted(rows,key=lambda r:-r['reference_directions'][k]['top_direction_absolute_cosine'])],
                        by_matrix_energy=[r['layer'] for r in sorted(rows,key=lambda r:-r['reference_directions'][k]['matrix_energy_along_direction'])]) for k in references},
        limitations=['Static weight overlap is a candidate-selection signal; no causal looping direction has been localized with activations.',
                     'Router measurements concern input sensitivity, shared-expert measurements concern output directions; their percentages are not interchangeable.',
                     'Routed samples cover 4/512 experts per layer and do not reveal which experts a looping prompt actually uses.',
                     'Cross-layer coordinate overlap does not establish a repeated functional role.',
                     'Layer45 original direction is already suppressed in the baseline; its current leading mode is a different direction.'],
        bytes_read=bytes_read,seconds=time.monotonic()-started,model_unchanged=True)
    n.harness.write(ROOT/'all_layer_target_map.json',report)
    np.savez_compressed(ROOT/'all_shared_top4_directions.npz',**directions)
    print(json.dumps({'coverage':report['coverage'],'top_l45_overlap':report['rankings']['removed_l45'],'seconds':report['seconds']}),flush=True)

if __name__=='__main__':main()
