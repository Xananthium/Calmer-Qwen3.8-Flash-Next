# Calmer Qwen3.8 Flash Next experiment

The selected release removes one fixed direction from the shared-expert down projections at zero-based layers 44 and 45. Its four-question natural-completion panel used **48,841 tokens versus 75,535 for unedited weights**, with four correct final answers in both arms. This report records the conditions under which that result was observed, including the edits that made responses longer.

## Model and intervention

The source is AgentionAI's `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`, pinned at revision `8efdc41d092da63763f39173b0fff80eac1c0898`. Its published SHA256 matches the locally recorded source hash: `7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f`. The file is 91,965,774,272 bytes and uses the `qwen4exp` architecture with custom mixed-precision Gyro tensor types.

Layer 45's original decoded Q8_0 shared-expert output matrix supplied its leading left singular vector `d45`. Full projection removal was applied at 45 and, using the same direction, at 44. The matrices retain Q8_0 encoding. Layer 45 changed by about 16.75% of its Frobenius norm; layer 44 by 10.20%. These percentages describe two matrices, not the whole model. After Q8 rounding, remaining amplitudes along the selected direction were approximately 0.100% at 45 and 0.195% at 44.

The release contains only 44+45. Layers 2 and 46 were restored to their original tensor bytes after the last experiment. Exact original/edited hashes and offsets are in [release_manifest.json](release/release_manifest.json). This same materialized Q8 representation was evaluated and restored for continued use; it is not a new F32 merged export of the older 27B sidecar experiments.

## Production protocol

Requests used the existing Strata backend through its reasoning router. The recorded engine version was 0.1.59.1. All natural-panel comparisons retained two RTX 3090s, temperature 1.3, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1, medium reasoning, 262,144 context, Q4_0 KV, four slots, layer split 24, four-token MTP speculation at minimum probability 0.7, and two pipeline windows.

The existing Huihui runtime control vector remained enabled in both arms: project mode, per-layer directions, inclusive layers 4–44, scale 1.0. Its SHA256 is `59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38`; its [upstream file](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF/blob/main/Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf) is separate from the released weight edits. No evaluation without that vector was run.

Natural-completion settings disabled answer-room reservation, reasoning-budget closure, close retry, loop recovery, and repeated-token stopping. A 75,000 output-token ceiling applied to the later test requests, not to global production. The 45-only natural reference had `max_tokens=-1`; all of its completed outputs ended below 75,000, so this ceiling difference did not truncate them. Requests used seed 20261118 except explicitly labeled second-seed cases. Later arms ran math first; early arms ran economics first. Shared production traffic, scheduling, cache state, and speculative decoding limit token-identical replay.

Natural completion required a received stream terminator, no error, `finish_reason=stop`, a nonempty final answer, and no inserted wrap marker. The observed wrap marker was “I have thought about this long enough; time to give my answer.” It appeared in earlier forced-ending experiments, which are not mixed into the natural panel.

The saved API model alias `qwen3.8-27b-hauhau-q4` is historical routing configuration. Process arguments, model paths, and tensor hashes identify the actual model as Flash-Next.

## Main-panel results

The four common cases are HLE economics ID `673722c82bfc8ab579ed111f`, HLE mathematics ID `6739e06f5a81384176dd28cd`, and GSM8K test rows 936 and 236. Their exact frozen requests, grades, and request identities are represented in the archive; HLE text is withheld under the public-export convention.

The [README comparison table](README.md#all-variants-side-by-side) includes every main-panel variant. [main_panel.csv](evidence/main_panel.csv) contains per-case tokens, completion state, waits, elapsed time, and lower-bound flags. [comparison.json](evidence/comparison.json) retains the original aggregate and configuration comparisons.

Relative to unedited weights, completed four-case totals changed by +5.89% for 45 only, −19.94% for 45+46, **−35.34% for 44+45**, and +13.53% for full layer-2 +44+45. The selected edit shortened math by 47.65% and economics by 7.31%, while the arithmetic controls together grew 14.08%.

The 44+45+46 math request was stopped at an observed count of at least 70,721 tokens without a final answer. The later layer-2 half-strength/layer-46 quarter-strength math request was stopped at at least 71,551 tokens, also without a final. These values are counters captured before cancellation, not final API usage. Neither is a completed wrong answer or an example of natural shortening. The other three main cases were not run for the partial-strength variant.

## Constructed loop and tool probes

Four simulated tool-recovery histories, two primed arithmetic continuations, and two clean arithmetic controls formed the eight-case short panel. All completed naturally. No emitted shell commands were executed.

| Variant | Tokens | Automatic correct |
|---|---:|---:|
| Unedited | 1,865 | 7/8 |
| 45 only | 2,017 | 8/8 |
| 45+46 | 1,635 | 6/8 |
| 44+45 | 1,796 | 8/8 |
| 44+45+46 | 1,480 | 7/8 |
| Full 2+44+45 | 1,593 | 8/8 |
| Half 2+44+45+quarter 46 | 1,534 | 8/8 |

The unedited Git case repeated `git diff` after wrong-directory feedback, even though its reasoning named the correction. The selected edit emitted `git -C /app/project diff`. The 45+46 automatic misses chose inspection actions (`ls /app` and `cat package.json`) instead of the requested direct corrections; they did not simply repeat the failing operation.

The triplet's Git output, `cd /app/project && git status && git diff --stat`, was a valid corrective action missed by the strict expected-command grader. Its original automatic 7/8 is preserved, with the separate manual review. These distinctions matter when interpreting a one-case score change. Inspect the exact request and output files under each variant's `loop_probe/` directory.

## Earlier budgets and additional seeds

The first 8,192-token paired study ran ten requests per arm under the then-current forced-ending settings: unedited scored 7/10 and 45-only 6/10. The 32,768-token follow-up scored 6/10 for 45-only without a matched unedited arm at that budget. Neither study establishes the later natural-completion effect.

The 45-only natural reference additionally completed second-seed economics and math requests at 6,007 and 55,127 tokens. The 45+46 second-seed economics stream was interrupted; its second-seed math was not run. The 44+45 second-seed economics response ended naturally at 8,863 tokens with the correct explicit choice, but the original parser marked it unparsed because of Markdown formatting. The automatic result and manual correction remain separate. Its second-seed math was interrupted when moving to the triplet. These unequal panels are excluded from the four-case aggregate.

## Layer inspection

Static inspection covered all 48 shared-expert down projections with exact SVD, all 48 router matrices, and 192 routed experts: indices 0, 127, 255, and 511 at each layer. It did not exhaustively inspect all 24,576 experts, capture live expert routing, or map every attention/state-space tensor.

The original layer-45 direction recurred most strongly in nearby layers 46 and 44, followed by 43, 41, and 42. Independent high-concentration candidates included layers 31, 33, 28, and 2. These are numerical candidates, not established causal hesitation features. See [all_layer_target_map.json](experiments/strata_l45plus_nearby_natural_20261118/all_layer_target_map.json) and the saved directions.

Layer 2's strongest mode accounted for about 3.2278% of the decoded shared matrix's squared norm. A direct comparison with the pinned official BF16 layer-2 tensor found 3.2279%, with direction cosine 0.9999987. An alternate quantization also retained the feature. This rules out its creation by the local Q8 packing step; it does not establish its training origin or behavioral role. The BF16 tensor, source range/hash, quantization comparison, and chronological reports are archived. The earlier quantization note predates the later BF16 comparison and is retained as written.

The loaded control vector contained 47 entries with identical payloads. The file matched the published upstream adapter. Reviewed Strata loader/kernel paths assigned each entry to its own layer slot; they did not sum all entries. The separate experimental speed-projection adapter had 47 distinct directions and was not the active file. [The control-vector audit](experiments/strata_readonly_20261118/findings.json) records this format investigation separately from the behavior experiments.

## Restoration and reproducibility

At the end of the session, 44+45 was restored, the layer-2 and layer-46 originals were checked, the engine loaded successfully, and the boot configuration pointed at the selected model. Strata and the reasoning router were enabled for startup, with user-service lingering enabled. No reboot was performed merely to test this setting. The [restoration record](experiments/strata_l2a050_l44_l45_l46a025_20261118/restore_best.json) preserves the observed state.

The first edit exposed a loader issue: using the copied full GGUF for `--native` while retaining the original full file for `--ple-gguf` caused duplicate split-shard metadata. Both paths were aligned to the same edited GGUF. Subsequent experiments checked changed/retained/restored tensor spans, file size/layout, config, and live loading. Full-file copy readback was discontinued after user feedback; manifests explicitly distinguish targeted checks from a full byte audit.

The publication uploader hashes the selected file as part of transfer. Its resulting identity is recorded in the release manifest. No additional inference tests are part of publication. [Reconstruction instructions](evidence/RECONSTRUCTION.md) distinguish portable read-only/patch tools from historical scripts that manage the original host's services.

## Limits

These are small, previously exposed, adaptively selected panels with limited seed replication and no independent grading panel. Final-choice correctness does not validate every step of a long explanation. The runtime control vector remained active. Vision and broader retention were not evaluated. Production timing is not a controlled throughput benchmark. Raw “wait” counts and numerical weight concentration are descriptive; neither measures anxiety nor proves a repetition circuit.

The next experiment remains undecided. The recorded failures and partial runs are retained so future work can compare against the actual evidence rather than only the selected release.
