#!/usr/bin/env python3
"""Compare the layer2 structural mode across locally available model artifacts."""
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
    root=n.ROOT
    paths=[Path(n.SOURCE),Path('/home/researcher/models/qwen38-flash-next-gyro-m-abl/Qwen3.8-Flash-Next-Gyro-M-abliterated-TQ2_0.gguf')]
    paths+=sorted(Path('/home/researcher/models/qwen38-flash-next-iq3s/IQ3_S').glob('*.gguf'))
    rows=[];reference={}
    for path in paths:
        g=GGUFFile(path)
        ts={t.name:t for t in g.tensors}
        for layer in [0,1,2,3,44,46]:
            name=f'blk.{layer}.ffn_down_shexp.weight'
            if name not in ts:continue
            t=ts[name]
            with path.open('rb') as f:f.seek(g.data_start+t.offset);raw=f.read(t.expected_bytes())
            w=dequantize(np.frombuffer(raw,dtype=np.uint8),t.type_name).reshape(tuple(reversed(t.shape))).astype(np.float64)
            assert np.isfinite(w).all()
            u,s,_=np.linalg.svd(w,full_matrices=False);energy=float(np.sum(w*w))
            row=dict(path=str(path),general_name=g.metadata.get('general.name'),layer=layer,tensor=name,dtype=t.type_name,
                encoded_sha256=hashlib.sha256(raw).hexdigest(),top4_energy_shares=(s[:4]**2/energy).tolist(),
                max_abs=float(np.max(np.abs(w))),rms=float(np.sqrt(energy/w.size)),
                max_row_energy_share=float(np.sum(w*w,axis=1).max()/energy),
                max_column_energy_share=float(np.sum(w*w,axis=0).max()/energy),
                top_left_vector_max_coordinate_energy=float(np.max(u[:,0]**2)))
            if str(path)==n.SOURCE:
                reference[layer]=(w,u[:,0],row['encoded_sha256'])
            else:
                rw,ru,rhash=reference[layer]
                row.update(relative_weight_difference=float(np.linalg.norm(w-rw)/np.linalg.norm(rw)),
                    top_direction_absolute_cosine_to_production=float(abs(u[:,0]@ru)),
                    same_encoded_bytes_as_production=rhash==row['encoded_sha256'])
            if t.type_name=='Q8_0':
                blocks=np.frombuffer(raw,dtype=np.dtype([('d','<f2'),('q','i1',(32,))]))
                scales=np.abs(blocks['d'].astype(np.float64));q=blocks['q']
                row['q8_diagnostics']=dict(block_count=len(blocks),zero_scales=int(np.count_nonzero(scales==0)),
                    scale_percentiles={str(p):float(np.percentile(scales,p)) for p in [0,50,90,99,99.9,100]},
                    quantized_extreme_fraction=float(np.mean(np.abs(q.astype(np.int16))>=127)),
                    negative128_fraction=float(np.mean(q==-128)),
                    assumed_nearest_rounding_error_frobenius_bound=float(np.sqrt(32*np.sum((scales/2)**2))/np.sqrt(energy)),
                    bound_assumptions='Per-entry rounding error <= half stored scale, no clipping, excludes scale rounding and any preprocessing. Diagnostic assumption, not an observed error against source weights.')
            rows.append(row)
    out=dict(at=datetime.now(timezone.utc).isoformat(),rows=rows,
        scope='Read-only CPU tensor comparison; no model loading or inference.',
        limitations=['No original BF16/FP16 tensor has been compared.','Alternate artifacts may differ in preprocessing or ablation as well as quantization.','Matching Q8 bytes across files is not an independent unquantized reference.'])
    n.harness.write(root/'layer2_quant_check.json',out)
    for row in rows:
        if row['layer']==2:print(json.dumps(row),flush=True)

if __name__=='__main__':main()
