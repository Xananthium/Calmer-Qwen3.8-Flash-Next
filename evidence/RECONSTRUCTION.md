# Reproduce and inspect

The archive preserves all eight Flash-Next experiment directories from 2026-11-18. Earlier 27B work belongs to the [Calmer V1/V2 repository](https://github.com/Xananthium/Qwen3.8-Weight-Space-Programming). Files are organized by their historical directory names so protocol, request, result, score, deployment, and patch records stay together.

## Check published measurements

From the repository root:

```bash
python3 tools/verify_archive.py
```

This checks each exported file against the archive manifest, parses public JSON, replays the main-panel token totals, and verifies the selected patch payload hashes. It uses only Python's standard library and makes no inference request. Source hashes refer to the original local files; public hashes refer to the exported files after documented substitutions and withholding. They are intentionally different where exports changed.

## Rebuild the selected GGUF

Download the pinned base:

```bash
hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF \
  Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf \
  --revision 8efdc41d092da63763f39173b0fff80eac1c0898 \
  --local-dir upstream
```

Apply the exact saved Q8 payloads to a new copy:

```bash
python3 tools/rebuild_selected.py \
  --base upstream/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf \
  --output Calmer-Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf
```

The tool requires a new output path and never changes services. It verifies the source size, both original tensor hashes, both patch hashes, and the resulting edited spans. Add `--check-only` to check without creating a model, or `--verify-base-sha256` for an optional full source hash before copying. Normal reconstruction trusts the completed copy and does not reread the entire output merely to audit copying. Only the exact pinned base is supported.

The complete prebuilt model is on [Hugging Face](https://huggingface.co/Xananthium/Calmer-Qwen3.8-Flash-Next). Downloading it is simpler than allocating another 92 GB copy.

## Runtime dependencies

Use the upstream custom Gyro-capable Strata/Agention runtime. The saved settings are in [tested_runtime.json](../release/tested_runtime.json); each experiment also contains the actual historical config and process identity. A compatible expert pack/tokenizer, MTP assets, and expert profile are separate runtime assets. The control vector is separately downloadable from its recorded upstream repository and is not incorporated into the weight edits. The upstream quantization card provides runtime preparation instructions.

Both `--native` and `--ple-gguf` must point to the same selected full GGUF in the tested native/PLE layout. The old API alias is not the model identity. The natural panel used medium reasoning and no forced ending controls, with a 75,000-token ceiling per later test request. Shared production load and speculative decoding prevent a guarantee of identical generations on replay.

## HLE and raw outputs

Following the earlier Calmer repository's public-export convention and the [HLE authors' redistribution request](https://huggingface.co/datasets/cais/hle), HLE prompts, answer keys, full reasoning/final traces, and raw event streams remain in the local research archive. The public export retains every run's IDs, automatic status, numerical metrics, timing, incomplete flags, and hashes. Withheld fields have explicit `*_withheld` records; withheld event streams have `.withheld.json` records. This is not a claim that a raw HLE transcript is publicly available.

Obtain HLE through its official dataset source under its access conditions. Case IDs and frozen request parameters identify the inputs. The original request messages are hashed as UTF-8 JSON with `sort_keys=True` and separators `(',', ':')`, using Python's default ASCII escaping. If you have an authorized local copy of the exact messages, restore a public request template with:

```bash
python3 tools/rehydrate_request.py \
  --template experiments/strata_unedited_natural_20261118/baseline/05.request.json \
  --messages authorized-messages.json \
  --output restored-request.json
```

The tool refuses messages whose hash differs from the frozen request and makes no inference request. Source traces are identified by source-file hashes; stochastic reruns cannot reconstruct a historical trace byte for byte. The historical `production_comparison.py` records the exact request-construction logic and prior dataset-record sources.

Non-HLE outputs and prompts are included. Published raw event streams use lossless gzip where present. Their request/result summaries can be read without decompression. No emitted tool command was executed during the evaluation.

## Historical code and export changes

The Python scripts under `experiments/` are preserved research code, including preparation, restart, deployment, interruption, evaluation, and analysis scripts. They encode the original machine's layout and may change services or weights if adapted and run. They are not portable setup commands. The tools under `tools/` are the portable publication utilities described above.

Home-directory paths and host names were replaced with generic research paths; settings, tensor bytes, numerical results, and timestamps were preserved. Compiled Python caches and the local encoder shared library are excluded. The archive manifest accounts for every source file and its disposition. Derived model tensor payloads retain the Qwen model license; full model weights are distributed on Hugging Face.
