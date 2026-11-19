<div align="center">

# Calmer Qwen Flash Next

**Qwen3.8 Flash Next Weight-Space Programming**<br>
An open experiment in changing reasoning through small, fixed weight edits.

[![Download Calmer](https://img.shields.io/badge/Hugging_Face-Download_Calmer-FFD21E?style=for-the-badge&logo=huggingface&logoColor=111827)](https://huggingface.co/Xananthium/Calmer-Qwen3.8-Flash-Next)
[![Read the research](https://img.shields.io/badge/Research-Results_%26_methods-7C3AED?style=for-the-badge)](REPORT.md)

[![Experimental release](https://img.shields.io/badge/Status-Experimental-D97706?style=flat-square)](REPORT.md#limits)
[![Weight edits](https://img.shields.io/badge/Selected_layers-44_%2B_45-0F766E?style=flat-square)](release/release_manifest.json)
[![Code license](https://img.shields.io/badge/Code-Apache_2.0-2563EB?style=flat-square)](LICENSE)
[![Model license](https://img.shields.io/badge/Weights-Qwen_Community_1.0-2563EB?style=flat-square)](licenses/QWEN-COMMUNITY-LICENSE-1.0.txt)

**[Research report](REPORT.md)** · **[Every experiment](experiments/README.md)** · **[Reproduce](evidence/RECONSTRUCTION.md)** · **[Earlier Calmer V1/V2](https://github.com/Xananthium/Qwen3.8-Weight-Space-Programming)**

</div>

---

**Calmer Qwen3.8 Flash Next · Gyro-M** applies fixed edits to the shared-expert down projections at **zero-based layers 44 and 45**. In a four-question exploratory comparison against the unedited weights, generated tokens fell **35.3%**, with **4/4 correct final answers and natural completions in both arms**. The owner selected 44+45 for continued production use. Layers 2 and 46 are original in the release.

Both arms ran through the same Strata production backend on two RTX 3090s, at **medium reasoning and temperature 1.3**, with the same existing runtime control vector. That vector is separate from the GGUF weight edits. These are reused development questions, not a full benchmark or an evaluation without runtime steering.

## Results at a glance

| Main-panel case | Unedited | Selected 44+45 | Change |
|---|---:|---:|---:|
| Economics | 21,315 | 19,756 | −7.3% |
| Mathematics | 53,084 | 27,789 | −47.7% |
| Hotel arithmetic | 581 | 657 | +13.1% |
| Pens arithmetic | 555 | 639 | +15.1% |
| **Generated tokens** | **75,535** | **48,841** | **−35.3%** |
| Correct final answers | 4/4 | 4/4 | — |
| Natural completions | 4/4 | 4/4 | — |
| Eight short loop/tool probes, automatic grade | 7/8 | 8/8 | — |

The main panel used a 75,000-token request ceiling, with forced thinking closure and loop recovery disabled. All four selected outputs ended on their own. The arithmetic controls became slightly longer. Raw “wait” occurrences fell from 115 to 54 across the four outputs; the count describes language, not an internal mental state.

## All variants, side by side

Counts include reasoning and final output. A dagger marks an interrupted run with a lower-bound token count, not a finished answer.

| Variant | Economics | Math | Hotel | Pens | Total | Completed correct |
|---|---:|---:|---:|---:|---:|---:|
| Unedited | 21,315 | 53,084 | 581 | 555 | 75,535 | 4/4 |
| 45 only | 12,899 | 65,855 | 492 | 741 | 79,987 | 4/4 |
| 45+46 | 23,380 | 35,760 | 581 | 753 | 60,474 | 4/4 |
| **44+45 — selected** | **19,756** | **27,789** | **657** | **639** | **48,841** | **4/4** |
| 44+45+46 | 18,390 | ≥70,721† | 526 | 570 | ≥90,207† | 3/4; one interrupted |
| 2(full)+44+45 | 21,482 | 62,984 | 737 | 554 | 85,757 | 4/4 |
| 2(0.5)+44+45+46(0.25) | Not run | ≥71,551† | Not run | Not run | Incomplete | One interrupted |

The 45-only natural run had no finite request ceiling; its outputs all stopped below 75,000. Other main comparisons used that ceiling. The final partial-strength variant completed all eight short probes correctly before its long math run was interrupted. Earlier 8,192/32,768-token trials used different ending controls and are preserved as separate studies. [Full conditions, second seeds, automatic-grade exceptions, and failures →](REPORT.md)

## A change you can read

In a constructed Git recovery prompt, three `git diff` attempts failed because the current directory was `/tmp`. The supplied feedback explicitly requested `git -C /app/project diff`.

| Model | Emitted next command | Result |
|---|---|---|
| Unedited | `git diff` | Repeated the failed action despite mentioning the correct fix in its reasoning |
| Selected 44+45 | `git -C /app/project diff` | Followed the correction |

These are simulated tool histories; the emitted commands were not executed. Read the [unedited output](experiments/strata_unedited_natural_20261118/loop_probe/baseline/L04.result.json), [selected output](experiments/strata_l44_l45_samedirection_20261118/loop_probe/edited/L04.result.json), and [exact prompt](experiments/strata_unedited_natural_20261118/loop_probe/baseline/L04.request.json).

## What changed

The leading left singular vector of the original layer-45 shared-expert down projection, `d45`, supplies both edits:

```text
W45_new = W45 - outer(d45, d45.T @ W45)
W44_new = W44 - outer(d45, d45.T @ W44)
```

Both strengths are 1.0. Layer 44 uses **layer 45's original direction**, not its own leading singular vector. The matrices are decoded, edited, and packed back into their original Q8_0 representation. The upstream Gyro-M model is mixed precision; the `TQ2_0` filename does not mean every tensor is two-bit. This exact materialized edit representation was evaluated and restored for production.

The direction came from weight SVD, not a fitted hesitation detector. Static inspection covered all 48 shared-expert down projections and routers, plus four routed experts per layer. A suspected layer-2 quantization artifact also appeared in the official BF16 tensor with almost identical direction and strength. [Layer inspection and provenance →](REPORT.md#layer-inspection)

## Pick a starting point

| You want to… | Start here |
|---|---|
| Try the selected model | [Hugging Face model card and GGUF](https://huggingface.co/Xananthium/Calmer-Qwen3.8-Flash-Next) |
| Compare every tested variant | [Report](REPORT.md), [main-panel CSV](evidence/main_panel.csv), [comparison JSON](evidence/comparison.json) |
| Read prompts and saved outputs | [Experiment directory index](experiments/README.md) |
| Inspect every published file's provenance | [Archive manifest](evidence/archive_manifest.json) |
| Rebuild the exact two tensor edits | [Reconstruction instructions](evidence/RECONSTRUCTION.md), [portable reconstruction tool](tools/rebuild_selected.py) |
| Extend the experiment | [Contributor guide](CONTRIBUTING.md) |

Verify the public archive without a model, GPU, or dataset access:

```bash
python3 tools/verify_archive.py
```

The archive accounts for all files in the eight Flash-Next experiment directories. It includes scores, configs, tensor patches, static analyses, historical scripts, non-HLE raw outputs, and unsuccessful/interrupted runs. Following the earlier Calmer repository's convention and the [HLE dataset card's request](https://huggingface.co/datasets/cais/hle), full HLE prompts and traces remain local; published hashes, measurements, IDs, and request templates preserve their provenance. Local compiled caches are excluded, and machine-specific home/host names are replaced. The manifest records each disposition.

## Interpretation and attribution

The panel was small, reused, adaptively explored, and largely one seed. Final-choice correctness is separate from proof validity. Shorter reasoning did not occur for every question or every edit. “Calmer” describes observed text behavior in these tests, not anxiety or wellbeing.

Prepared by **Xananthium with AI assistance**, including code, analysis, and publication. Credit to **Qwen/Alibaba Cloud**, **AgentionAI**, the **Strata** runtime contributors, and the **HLE/GSM8K** authors. Original project code and documentation are [Apache-2.0](LICENSE); model-derived tensor files retain the [Qwen Community License 1.0](licenses/QWEN-COMMUNITY-LICENSE-1.0.txt). See [NOTICE.txt](NOTICE.txt).
