# Contributing

Useful follow-ups include additional seeds, fresh questions, broader retention checks, measurements with and without the separate runtime control vector, and closer reviews of long explanations. Negative results and incomplete runs belong in the record.

For a new experiment, identify the exact model file and tensor hashes, runtime version/assets, prompt identity, sampling settings, stopping controls, seed, and budget. Keep final-answer correctness, explanation quality, and natural completion separate. Preserve automatic grades when adding a manual correction. Mark canceled counters as lower bounds rather than completed outputs.

Use the public archive checker before proposing changes:

```bash
python3 tools/verify_archive.py
```

Original code and documentation use Apache-2.0. Model-derived tensor payloads retain the Qwen Community License 1.0. Follow the archive's HLE publication convention and document any export changes. The files under `experiments/` are historical host-management scripts; inspect them before adapting them to another machine.
