#!/usr/bin/env python3
"""Reconstruct the selected Calmer GGUF from the exact upstream GGUF and saved patches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

def sha(data): return hashlib.sha256(data).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--release-dir',type=Path,default=Path(__file__).resolve().parents[1]/'release')
    parser.add_argument('--verify-base-sha256',action='store_true',help='Optional full source read before copying; otherwise verify size and both source tensor spans.')
    parser.add_argument('--check-only',action='store_true',help='Validate source tensor and patch identities without creating a model.')
    a=parser.parse_args()
    m=json.loads((a.release_dir/'release_manifest.json').read_text())
    if a.base.resolve()==a.output.resolve() or a.output.exists():
        parser.error('Output must be a new path, different from the base model.')
    original_stat=a.base.stat()
    if original_stat.st_size!=m['bytes']: raise ValueError('Unexpected base size')
    payloads=[]
    with a.base.open('rb') as src:
        for edit in m['edits']:
            src.seek(edit['offset']); b=src.read(edit['bytes'])
            if sha(b)!=edit['original_sha256']: raise ValueError(f"Base tensor mismatch: layer {edit['layer']}")
            patch=(a.release_dir/edit['patch']).read_bytes()
            if len(patch)!=edit['bytes'] or sha(patch)!=edit['edited_sha256']: raise ValueError('Patch identity mismatch')
            payloads.append((edit,patch))
    if a.verify_base_sha256:
        h=hashlib.sha256()
        with a.base.open('rb') as src:
            while chunk:=src.read(16*1024*1024): h.update(chunk)
        if h.hexdigest()!=m['source']['sha256']: raise ValueError('Base full SHA256 mismatch')
    if a.check_only:
        print('Source spans and release patches match; no model created.'); return
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects an existing model. Copy is trusted; only patched spans are read back.
    with a.output.open('xb') as dst, a.base.open('rb') as src:
        shutil.copyfileobj(src,dst,length=16*1024*1024)
        for edit,patch in payloads:
            dst.seek(edit['offset']); dst.write(patch)
        dst.flush(); os.fsync(dst.fileno())
    if a.base.stat().st_mtime_ns!=original_stat.st_mtime_ns: raise RuntimeError('Base changed during copying')
    if a.output.stat().st_size!=m['bytes']: raise RuntimeError('Output size mismatch')
    with a.output.open('rb') as dst:
        for edit,_ in payloads:
            dst.seek(edit['offset'])
            if sha(dst.read(edit['bytes']))!=edit['edited_sha256']: raise RuntimeError('Edited span mismatch')
    print(f'Created {a.output}; selected tensor hashes match. No production service was changed.')

if __name__=='__main__': main()
