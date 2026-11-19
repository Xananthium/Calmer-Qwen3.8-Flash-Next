#!/usr/bin/env python3
"""Materialize one strong, reversible shared-expert edit in a copied GGUF."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['STRATA_GGUF_PY'] = '/home/researcher/strata-122b/llama.cpp/gguf-py'

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45_svd1_20261118')
STRATA = Path('/home/researcher/strata-122b')
SOURCE = Path('/home/researcher/models/qwen38-flash-next-gyro-m/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf')
TARGET = Path('/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l45-svd1/Qwen3.8-Flash-Next-Gyro-M-Calmer-L45-SVD1-TQ2_0.gguf')
TENSOR = 'blk.45.ffn_down_shexp.weight'
CONFIG_NAME = 'g4x-l45svd1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    start = time.time()
    sys.path.insert(0, str(STRATA/'src/tools'))
    from gguf_reader import GGUFFile
    from iq_pack import dequantize
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q, quants
    ROOT.mkdir(parents=True, exist_ok=True)
    baseline_scores = json.loads((ROOT/'baseline/scores.json').read_text())
    assert baseline_scores['complete'] and len(baseline_scores['rows']) == 10
    assert (ROOT/'baseline/identity_after.json').is_file()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix('.gguf.partial')
    candidate_config = STRATA/'fn-iq3s'/f'cfg-{CONFIG_NAME}.json'
    assert not TARGET.exists() and not temporary.exists() and not candidate_config.exists()
    original_stat = SOURCE.stat()
    assert shutil.disk_usage(TARGET.parent).free > original_stat.st_size + 10*1024**3
    for source, dest in [('cfg-g4x.json', 'baseline_config.json'),
                         ('boot.env', 'baseline_boot.env'), ('.serving', 'baseline_serving.txt')]:
        shutil.copyfile(STRATA/'fn-iq3s'/source, ROOT/dest)
    baseline = json.loads((ROOT/'baseline_config.json').read_text())
    assert baseline['args'][baseline['args'].index('--native')+1] == str(SOURCE)
    g = GGUFFile(SOURCE)
    t = next(t for t in g.tensors if t.name == TENSOR)
    assert t.type_name == 'Q8_0' and t.shape == [640, 2560]
    offset, nbytes = g.data_start+t.offset, t.expected_bytes()
    with SOURCE.open('rb') as f:
        f.seek(offset)
        original = f.read(nbytes)
    assert sha(original) == '39ddadd3fdd10fd58b32f6e8d8433feb987406eb96535445f0c8eb8346ca2c3b'
    w = dequantize(np.frombuffer(original, dtype=np.uint8), 'Q8_0').reshape(2560,640).astype(np.float64)
    u, singular, _ = np.linalg.svd(w, full_matrices=False)
    direction = u[:, 0]
    coeff = direction @ w
    edited = w - np.outer(direction, coeff)
    packed = quants.quantize(edited.astype(np.float32), Q.Q8_0).tobytes()
    assert len(packed) == nbytes and packed != original
    decoded = dequantize(np.frombuffer(packed, dtype=np.uint8), 'Q8_0').reshape(w.shape).astype(np.float64)
    assert np.isfinite(decoded).all()
    remaining = np.linalg.norm(direction @ decoded)/np.linalg.norm(coeff)
    assert remaining < 0.02, remaining
    np.savez_compressed(ROOT/'edit.npz', direction=direction, coeff=coeff, singular_values=singular, alpha=np.array(1.0))
    (ROOT/'original_tensor.q8').write_bytes(original)
    (ROOT/'edited_tensor.q8').write_bytes(packed)
    result = dict(candidate='l45-shared-svd1-alpha1-q8', source=str(SOURCE), target=str(TARGET),
                  tensor=TENSOR, offset=offset, tensor_bytes=nbytes, shape=[2560,640],
                  formula='W_new = W - outer(u, u.T @ W); u is the exact leading left singular vector of decoded W',
                  alpha=1.0, scope='Only layer 45 shared-expert down projection; original Q8_0 encoding retained',
                  hypothesis='Strong spectral deflation trial, not a fitted repetition-specific direction',
                  original_tensor_sha256=sha(original), edited_tensor_sha256=sha(packed),
                  original_top_singular_value=float(singular[0]),
                  original_top_mode_energy_share=float(singular[0]**2/np.sum(w*w)),
                  relative_frobenius_change=float(np.linalg.norm(decoded-w)/np.linalg.norm(w)),
                  top_mode_fraction_remaining_after_q8=float(remaining),
                  q8_rounding_relative_to_ideal=float(np.linalg.norm(decoded-edited)/np.linalg.norm(edited)),
                  baseline_config=str(ROOT/'baseline_config.json'), candidate_config=str(candidate_config),
                  changed_runtime_arguments=['--native'], other_settings_unchanged=True,
                  rollback_config_name='g4x', baseline_generation_run=True,
                  baseline_scores=str(ROOT/'baseline/scores.json'),
                  protocol_sha256=sha((ROOT/'protocol.json').read_bytes()))
    print(json.dumps({'stage':'tensor_prepared', **result}), flush=True)

    # Stream a complete copy, replacing only this tensor's byte interval. Hash
    # both streams, then independently read the result to verify the full copy.
    hs, ht = hashlib.sha256(), hashlib.sha256()
    position, last_report = 0, time.monotonic()
    with SOURCE.open('rb') as src, temporary.open('xb') as dst:
        while chunk := src.read(32*1024*1024):
            hs.update(chunk)
            end = position+len(chunk)
            lo, hi = max(position,offset), min(end,offset+nbytes)
            if lo < hi:
                chunk = chunk[:lo-position]+packed[lo-offset:hi-offset]+chunk[hi-position:]
            ht.update(chunk)
            dst.write(chunk)
            position = end
            if time.monotonic()-last_report >= 12:
                print(json.dumps({'stage':'copying','GiB':round(position/1024**3,2),'percent':round(position/original_stat.st_size*100,1)}),flush=True)
                last_report=time.monotonic()
        dst.flush()
        os.fsync(dst.fileno())
    assert position == original_stat.st_size
    assert SOURCE.stat().st_size == original_stat.st_size and SOURCE.stat().st_mtime_ns == original_stat.st_mtime_ns
    verify = hashlib.sha256()
    print(json.dumps({'stage':'verifying_full_candidate'}), flush=True)
    with temporary.open('rb') as f:
        while chunk := f.read(32*1024*1024):
            verify.update(chunk)
    assert verify.digest() == ht.digest()
    temporary.rename(TARGET)
    g2 = GGUFFile(TARGET)
    assert g.metadata == g2.metadata and g.tensors == g2.tensors
    assert g.data_start == g2.data_start
    result.update(source_sha256=hs.hexdigest(), candidate_sha256=ht.hexdigest(),
                  full_candidate_hash_verified=True, header_and_tensor_layout_unchanged=True,
                  unchanged_bytes_copied=position-nbytes, preparation_seconds=time.time()-start,
                  status='prepared_not_deployed')
    candidate = json.loads(json.dumps(baseline))
    candidate['args'][candidate['args'].index('--native')+1] = str(TARGET)
    candidate_config.write_text(json.dumps(candidate,indent=2)+'\n')
    (ROOT/'candidate_config.json').write_text(json.dumps(candidate,indent=2)+'\n')
    assert [i for i,(x,y) in enumerate(zip(baseline['args'],candidate['args'])) if x!=y] == [baseline['args'].index('--native')+1]
    check=json.loads(json.dumps(candidate))
    check['args'][check['args'].index('--native')+1]=str(SOURCE)
    assert check == baseline
    (ROOT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
