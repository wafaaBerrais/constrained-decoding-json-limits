# Testing the extracted rules with a real LLM (XGrammar)

The internship pipeline replays known instances through each framework's token mask; no model generates anything. This folder is a small follow-up: a real LLM generates under **XGrammar** constrained decoding, and we check whether the decision-tree rules in [`../coverage_prediction/rules/xgr/`](../coverage_prediction/rules/xgr/) point to the cases where generation actually goes wrong.

It is a **pilot on 60 cases with a 0.5B model on CPU**, meant to check the approach, not to give statistically solid numbers.

## Setup

| | |
|---|---|
| Framework | XGrammar 0.1.17, configured like MaskBench's `xgr` engine (`strict_mode=True`, `any_whitespace=False`) |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, local, CPU, greedy decoding |
| Rules | positive leaves of the XGrammar OVER and UNDER trees (`*_positive_rules.csv`) |
| OVER cases | valid instances from **Kubernetes** (never seen when the rules were extracted) |
| UNDER cases | invalid instances from the **held-out GitHub test split** (XGrammar makes no UNDER error on Kubernetes) |
| Sample | 15 cases flagged by a rule + 15 unflagged controls per error type, one test per schema, instance ≤ 120 tokens |

## Method: a copy task

The model is asked to return a known instance as JSON, with the schema enforced by XGrammar.

| Error type | Instance given to the model | What counts as an error |
|---|---|---|
| **OVER** | valid | the output is not the instance (XGrammar blocked a valid output) |
| **UNDER** | invalid | the output fails `jsonschema` validation (XGrammar let an invalid output through) |

A **rule-guided mitigation** is then applied to the flagged cases only:

- OVER risk: generate **without** the grammar, validate with `jsonschema`, retry once on failure.
- UNDER risk: keep the grammar, **add the schema to the prompt**, validate with `jsonschema`, retry once with the validator's message.

## Results

| Error type | Group | n | Errors at mask level | Errors in generation (baseline) | Errors after mitigation |
|---|---|---:|---:|---:|---:|
| OVER | flagged | 15 | 15 | **15** | 14 |
| OVER | control | 15 | 1 | 4 | – |
| UNDER | flagged | 15 | 9 | **8** | 7 |
| UNDER | control | 15 | 1 | 2 | – |

"Errors at mask level" redoes the internship check with Qwen's tokenizer; it matches the internship labels (Llama 3.1 tokenizer) on all 60 cases.

What this shows:

- **The rules carry over to real generation.** Flagged cases fail far more often than controls: 15/15 vs 4/15 for OVER, 8/15 vs 2/15 for UNDER.
- **Small-model noise is visible in the controls.** The 4 OVER control errors are copy mistakes by the model (a flipped boolean, two swapped array items, dropped keys), on instances XGrammar accepts. The single control that XGrammar rejects at mask level was still reproduced correctly.
- **OVER mitigation gives valid output, but not the exact instance.** Without the grammar, all 15 outputs are schema-valid. Only 1 equals the instance exactly, because every flagged Kubernetes instance contains an extra key with a trailing space (for example `"fsType "`), which the schema tolerates, XGrammar's strict mode forbids, and the model silently rewrites as `"fsType"`. Ignoring whitespace in keys, 13/15 match.
- **Prompt enrichment does not fix UNDER with a 0.5B model.** The model keeps copying the invalid value (`0 is less than the minimum of 1`, `'invalid_email' is not a 'email'`), even after seeing the validator's message. The `jsonschema` check does catch all 7 remaining invalid outputs, so they are detected instead of being returned silently.

## Limitations

- 60 cases, one small model, greedy decoding, one framework.
- The rules use instance features (for example "the value is outside the bound"). The copy task knows the instance in advance; in free generation only the schema-side conditions would be available.
- The modeling tables must have correct test numbers, because instances are reloaded from `test_index`. A few rows had a wrong one; they were repaired with `scripts/fix_modeling_table_test_ids.py`, and sampling still skips any row whose expected validity disagrees with the benchmark.

## Reproduce

```bash
./setup_benchmark.sh                      # from the repository root: fetches data/ and maskbench/
pip install xgrammar==0.1.17 torch transformers jsonschema matplotlib

cd extension_jsonschemabench/llm_validation
python run_llm_validation.py sample       # -> sample.jsonl
python run_llm_validation.py run          # -> results.jsonl (resumable, about 20 minutes on a laptop CPU)
python run_llm_validation.py report       # -> summary.csv, summary.md
python plot_llm_validation.py             # -> docs/figures/llm_validation_flagged_vs_control.svg
```

`results.jsonl` holds every prompt outcome (generated text, validity, validator message) for each case.
