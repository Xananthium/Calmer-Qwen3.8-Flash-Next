#!/usr/bin/env python3
"""CPU-only static inspection of the deployed Strata artifact; no serving edits.

Reuses the original Calmer inspect_weights.py spectral method and Strata's
existing GGUF reader and CPU decoders. No inference or GPU imports/calls.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import collections
import hashlib
import json
import pathlib
import resource
import sys
import time
import urllib.request

import numpy as np


def identity(path):
    s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns,
                device=s.st_dev, inode=s.st_ino)


def health():
    with urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=10) as r:
        return json.load(r)


def process_identity(pid):
    p = pathlib.Path('/proc') / str(pid)
    # Field 22 follows the parenthesized comm; starttime is index 19 after it.
    return dict(pid=pid, start_ticks=p.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19],
                exe=os.readlink(p/'exe'),
                argv=p.joinpath('cmdline').read_bytes().decode().split('\0')[:-1])


def stats(w, spectral=False):
    finite = np.isfinite(w)
    out = dict(shape=list(w.shape), nonfinite=int(w.size-finite.sum()),
               zeros=int(np.count_nonzero(w == 0)))
    if not finite.all():
        return out, None
    energy = float(np.sum(w.astype(np.float64)**2))
    out.update(min=float(w.min()), max=float(w.max()),
               rms=float(np.sqrt(energy/w.size)), frobenius_norm=float(np.sqrt(energy)))
    if w.ndim != 2 or not spectral or energy == 0:
        return out, None
    columns = np.sum(w.astype(np.float64)**2, axis=0)
    ordered = np.sort(columns)
    out['top64_column_energy_share'] = float(ordered[-64:].sum()/energy)
    out['top1pct_column_energy_share'] = float(ordered[-max(1, int(np.ceil(w.shape[1]*0.01))):].sum()/energy)
    rng = np.random.default_rng(20261118)
    q, _ = np.linalg.qr(rng.standard_normal((w.shape[0], 4)).astype(np.float32))
    for _ in range(5):
        q, _ = np.linalg.qr(w @ (w.T @ q))
    u, singular, _ = np.linalg.svd(q.T @ w, full_matrices=False)
    out['approx_principal_energy_shares'] = (singular.astype(np.float64)**2/energy).tolist()
    out['spectral_method'] = 'randomized rank-4; five power iterations; seed 20261118'
    return out, (q @ u).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strata', type=pathlib.Path, required=True)
    ap.add_argument('--model', type=pathlib.Path, required=True)
    ap.add_argument('--cvec', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--tq-lib', type=pathlib.Path, required=True)
    a = ap.parse_args()
    os.environ['TQENC_LIB'] = str(a.tq_lib)
    os.environ['STRATA_GGUF_PY'] = str(a.strata.parent/'llama.cpp/gguf-py')
    sys.path.insert(0, str(a.strata/'tools'))
    from gguf_reader import GGUFFile
    from iq_pack import dequantize
    import tqenc

    a.out.mkdir(parents=True, exist_ok=True)
    begin = time.time()
    initial_process, initial_health = process_identity(a.pid), health()
    before = identity(a.model)
    g = GGUFFile(a.model)
    by_name = {t.name: t for t in g.tensors}
    arch = g.metadata['general.architecture']
    count = g.metadata[arch+'.block_count']
    expert_count = g.metadata[arch+'.expert_count']
    sample_experts = [0, 127, 255, 511]
    assert count == 48 and expert_count == 512
    assert all(0 <= e < expert_count for e in sample_experts)
    records, directions, read_bytes = [], {}, 0
    with a.model.open('rb') as f:
        header_hash = hashlib.sha256(f.read(g.data_start)).hexdigest()
        read_bytes += g.data_start

        def read(t, expert=None):
            nonlocal read_bytes
            size = t.expected_bytes()
            assert size is not None
            shape = t.shape
            offset = g.data_start+t.offset
            if expert is not None:
                assert len(shape) == 3 and size % shape[2] == 0
                size //= shape[2]
                offset += expert*size
                shape = shape[:2]
            f.seek(offset)
            raw = f.read(size)
            assert len(raw) == size
            read_bytes += size
            packed = np.frombuffer(raw, dtype=np.uint8)
            if t.type_name == 'TQ2_T':
                values = tqenc.decode(packed, tqenc.TQ2_T)
            else:
                values = dequantize(packed, t.type_name)
            w = values.reshape(tuple(reversed(shape)))
            return w, hashlib.sha256(raw).hexdigest()

        for layer in range(count):
            t = by_name[f'blk.{layer}.ffn_gate_inp.weight']
            w, digest = read(t)
            router, _ = stats(w)
            row_norms = np.linalg.norm(w.astype(np.float64), axis=1)
            router.update(row_norm_min=float(row_norms.min()),
                          row_norm_median=float(np.median(row_norms)),
                          row_norm_max=float(row_norms.max()))
            records.append(dict(layer=layer, role='router', tensor=t.name,
                                dtype=t.type_name, encoded_sha256=digest, **router))
            todo = [(by_name[f'blk.{layer}.ffn_down_shexp.weight'], None)]
            todo += [(by_name[f'blk.{layer}.ffn_down_exps.weight'], e) for e in sample_experts]
            for t, expert in todo:
                w, digest = read(t, expert)
                result, vectors = stats(w, spectral=True)
                role = 'shared_down' if expert is None else 'routed_down_sample'
                records.append(dict(layer=layer, role=role, expert=expert,
                                    tensor=t.name, dtype=t.type_name,
                                    input_basis='native' if expert is None else 'stored Hadamard basis',
                                    encoded_sha256=digest, **result))
                if vectors is not None:
                    directions[f'layer_{layer}_{role}_{expert}'] = vectors
            if layer % 8 == 7:
                print(json.dumps(dict(completed_layers=layer+1, seconds=round(time.time()-begin, 2))), flush=True)

    c = GGUFFile(a.cvec)
    cvec = []
    with a.cvec.open('rb') as f:
        for t in c.tensors:
            assert t.type_name == 'F32'
            f.seek(c.data_start+t.offset)
            raw = f.read(t.expected_bytes())
            v = np.frombuffer(raw, dtype='<f4')
            row, _ = stats(v)
            cvec.append(dict(name=t.name, **row))

    inventory = [dict(name=t.name, shape=t.shape, dtype=t.type_name,
                      absolute_offset=g.data_start+t.offset, bytes=t.expected_bytes()) for t in g.tensors]
    selected_meta = {k:v for k,v in g.metadata.items()
                     if not k.startswith('tokenizer.') and not isinstance(v, (list, dict))}
    after = identity(a.model)
    final_process, final_health = process_identity(a.pid), health()
    report = dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        scope='static CPU-only scan; all 48 routers and shared down matrices; four fixed routed down experts per layer',
        model_before=before, model_after=after, model_header_sha256=header_hash,
        process_before=initial_process, process_after=final_process,
        health_before=initial_health, health_after=final_health,
        metadata=selected_meta, tensor_types=g.by_type(), sample_experts=sample_experts,
        bytes_read_from_model=read_bytes, seconds=time.time()-begin,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        checks=dict(same_model_stat=before == after, same_engine_process=initial_process == final_process,
                    source_tensors_scanned=len(records), scanned_nonfinite=sum(r['nonfinite'] for r in records),
                    cvec_nonfinite=sum(r['nonfinite'] for r in cvec),
                    model_loaded_before=initial_health.get('loaded'), model_loaded_after=final_health.get('loaded')),
        rows=records, control_vectors=dict(path=str(a.cvec), metadata=c.metadata, rows=cvec),
        interpretation=[
            'Spectral structure is a hypothesis source, not evidence of anxiety or a repetition mechanism.',
            'Routed samples are expert IDs 0,127,255,511; not an exhaustive expert scan or a routing-usage measurement.',
            'Routed column energy is measured in the stored Hadamard basis; it does not identify unrotated neuron columns.',
            'Singular values and left singular subspace are invariant to an exact orthogonal input-basis transform.',
            'No live activations, expert routing events, prompt captures, or generation trials were collected.',
            'No live weights, service settings, model files, or source files were edited; only new inspection artifacts were written.',
            'Published Calmer V2 layers are 18,49,50,51 in a 64-block dense model; depth mapping alone does not transfer its learned vectors.',
            'The active model uses an additional per-layer projection control vector; future matched comparisons must record its state.',
        ],
    )
    (a.out/'tensor_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    (a.out/'inspection.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(a.out/'spectral_hypotheses.npz', **directions)
    shared = [r for r in records if r['role']=='shared_down']
    ranked = sorted(shared, key=lambda r:r['approx_principal_energy_shares'][0], reverse=True)
    print(json.dumps(dict(checks=report['checks'], seconds=report['seconds'],
                          peak_rss_mib=report['peak_rss_kib']/1024,
                          bytes_read=read_bytes, top_shared_by_first_spectral_share=[
                              dict(layer=r['layer'], share=r['approx_principal_energy_shares'][0])
                              for r in ranked[:8]], output=str(a.out)), indent=2), flush=True)


if __name__ == '__main__':
    main()
