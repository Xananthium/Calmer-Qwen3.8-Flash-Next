#!/usr/bin/env python3
"""Read just one pinned upstream tensor using bounded HTTP range requests."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['STRATA_GGUF_PY']='/home/researcher/strata-122b/llama.cpp/gguf-py'
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
from urllib.request import Request,urlopen
import numpy as np
import natural_panel as n

REV='812d96e10a0dd82b69d749e68693698cf6c36daa'
KEY='model.language_model.layers.2.mlp.shared_expert.down_proj.weight'

def get_range(url,start,length):
    assert 0<length<16000000
    req=Request(url+f'?download=true&tensor_range={start}-{start+length-1}',headers={'Range':f'bytes={start}-{start+length-1}'})
    with urlopen(req,timeout=40) as response:
        assert response.status==206, ('Server did not honor range; refusing full shard',response.status)
        content_range=response.headers.get('Content-Range','')
        assert content_range.startswith(f'bytes {start}-{start+length-1}/'),content_range
        raw=response.read(length+1)
    assert len(raw)==length
    return raw

def main():
    root=n.ROOT
    index=json.loads((root/'upstream_safetensors_index.json').read_text())
    shard=index['weight_map'][KEY]
    url=f'https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{REV}/{shard}'
    length=struct.unpack('<Q',get_range(url,0,8))[0]
    header=json.loads(get_range(url,8,length));entry=header[KEY]
    assert entry['shape']==[2560,640]
    start,end=entry['data_offsets']
    raw=get_range(url,8+length+start,end-start)
    (root/('upstream_layer2.'+entry['dtype'].lower())).write_bytes(raw)
    if entry['dtype']=='BF16':w=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view('<f4')
    elif entry['dtype']=='F16':w=np.frombuffer(raw,dtype='<f2')
    elif entry['dtype']=='F32':w=np.frombuffer(raw,dtype='<f4')
    else:raise ValueError(entry['dtype'])
    w=w.reshape(entry['shape']).astype(np.float64)
    assert np.isfinite(w).all()
    u,s,_=np.linalg.svd(w,full_matrices=False)
    sys.path.insert(0,'/home/researcher/strata-122b/src/tools')
    from gguf_reader import GGUFFile
    from iq_pack import dequantize
    g=GGUFFile(n.SOURCE);t=next(t for t in g.tensors if t.name=='blk.2.ffn_down_shexp.weight')
    with open(n.SOURCE,'rb') as f:f.seek(g.data_start+t.offset);packed=f.read(t.expected_bytes())
    q=dequantize(np.frombuffer(packed,dtype=np.uint8),t.type_name).reshape(w.shape).astype(np.float64)
    qu,qs,_=np.linalg.svd(q,full_matrices=False)
    result=dict(at=datetime.now(timezone.utc).isoformat(),upstream_repo='Qwen/Qwen3.8-Flash-Next',revision=REV,shard=shard,
        source_url=url,tensor=KEY,metadata=entry,tensor_sha256=hashlib.sha256(raw).hexdigest(),bytes_downloaded=8+length+len(raw),
        source_top4_energy_shares=(s[:4]**2/np.sum(w*w)).tolist(),production_top4_energy_shares=(qs[:4]**2/np.sum(q*q)).tolist(),
        production_relative_error=float(np.linalg.norm(q-w)/np.linalg.norm(w)),
        top_direction_absolute_cosine=float(abs(u[:,0]@qu[:,0])),
        original_top_direction_energy_in_production=float(np.sum((u[:,0]@q)**2)/np.sum(q*q)),
        source_matrix_nonfinite=0,scope='One tensor from the pinned official higher-precision release; only bounded ranges downloaded. Production unchanged.')
    n.harness.write(root/'upstream_layer2_comparison.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
