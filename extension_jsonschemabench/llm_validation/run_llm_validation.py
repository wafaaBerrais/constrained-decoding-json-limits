#!/usr/bin/env python3
"""Small end-to-end check of the extracted XGrammar rules with a real LLM.

The internship pipeline replays known instances through the framework's token
mask. Here a small local LLM actually generates under XGrammar constrained
decoding, on a "copy" task: the model is asked to output a known instance as
JSON under the schema.

- OVER  (valid instance, Kubernetes): if XGrammar is too restrictive, the model
  cannot reproduce the instance.
- UNDER (invalid instance, GitHub held-out split): if XGrammar is too
  permissive, the invalid instance goes through and the output fails
  `jsonschema` validation.

Cases flagged by the decision-tree rules are compared with unflagged controls,
then a rule-guided mitigation is applied to the flagged cases only.

Steps (each one is resumable):
    python run_llm_validation.py sample    # pick the cases -> sample.jsonl
    python run_llm_validation.py run       # generate      -> results.jsonl
    python run_llm_validation.py report    # aggregate     -> summary.csv / summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

csv.field_size_limit(1024 * 1024 * 1024)

ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = ROOT / "extension_jsonschemabench"
OUT_DIR = Path(__file__).resolve().parent
COVERAGE = EXT_ROOT / "coverage_prediction"
RULES_DIR = COVERAGE / "rules" / "xgr"
TESTS_DIR = ROOT / "maskbench" / "data"

# Where the candidate tests and their features come from, per error type.
FEATURE_TABLES = {
    "over": COVERAGE / "external_eval" / "Kubernetes" / "xgr" / "Kubernetes_external_features.csv",
    "under": COVERAGE / "modeles_predictifs" / "xgr" / "modeling" / "under_dataset.csv",
}
EXPECTED_VALIDITY = {"over": "valid", "under": "invalid"}

# One-hot encoded in the rule trees: `<column>_<value>`.
CATEGORICAL_FEATURES = ["object_additional_properties_case", "additionalProperties_mode"]

SYSTEM_PROMPT = "You are a data formatting assistant. You answer with a single JSON object and nothing else."


# --------------------------------------------------------------------------- rules


def load_rules(target: str) -> list[dict[str, Any]]:
    rules = []
    with open(RULES_DIR / target / f"{target}_positive_rules.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            conditions = []
            for part in row["rule"].split(" AND "):
                feature, operator, threshold = re.fullmatch(r"(\S+) (<=|>) (\S+)", part.strip()).groups()
                conditions.append((feature, operator, float(threshold)))
            rules.append({"leaf_id": row["leaf_id"], "conditions": conditions})
    return rules


def feature_value(row: dict[str, str], feature: str) -> float:
    if feature in row:
        raw = row[feature]
        if raw in ("true", "True"):
            return 1.0
        if raw in ("false", "False", "", "None"):
            return 0.0
        return float(raw)
    for column in CATEGORICAL_FEATURES:
        if feature.startswith(column + "_"):
            return 1.0 if row.get(column) == feature[len(column) + 1 :] else 0.0
    raise KeyError(feature)


def fired_rules(row: dict[str, str], rules: list[dict[str, Any]]) -> list[str]:
    fired = []
    for rule in rules:
        if all(
            feature_value(row, feature) <= threshold if operator == "<=" else feature_value(row, feature) > threshold
            for feature, operator, threshold in rule["conditions"]
        ):
            fired.append(rule["leaf_id"])
    return fired


# --------------------------------------------------------------------------- engine


class Engine:
    """Local Hugging Face model + XGrammar, configured like MaskBench's `xgr` engine."""

    def __init__(self, model_id: str, load_model: bool = True):
        import torch
        import xgrammar as xgr
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.xgr = xgr
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=config.vocab_size)
        self.compiler = xgr.GrammarCompiler(self.tokenizer_info, max_threads=1)
        self.bitmask = xgr.allocate_token_bitmask(1, self.tokenizer_info.vocab_size)
        self.stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        self.stop_ids.discard(None)
        self.model = None
        if load_model:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
            self.model.eval()

    def compile(self, schema: dict) -> Any:
        return self.compiler.compile_json_schema(json.dumps(schema), any_whitespace=False, strict_mode=True)

    def mask_accepts(self, grammar: Any, text: str) -> bool:
        """Same decision as the internship harness: every token of `text` passes the mask."""
        matcher = self.xgr.GrammarMatcher(grammar)
        return all(matcher.accept_token(token) for token in self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, user_prompt: str, grammar: Any | None, max_new_tokens: int) -> dict[str, Any]:
        """Greedy decoding, constrained by `grammar` when given."""
        torch = self.torch
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        matcher = self.xgr.GrammarMatcher(grammar) if grammar is not None else None
        generated: list[int] = []
        finished = False
        past = None
        started = time.monotonic()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = self.model(input_ids=input_ids, past_key_values=past, use_cache=True)
                past = output.past_key_values
                logits = output.logits[:, -1, :]
                if matcher is not None:
                    matcher.fill_next_token_bitmask(self.bitmask)
                    self.xgr.apply_token_bitmask_inplace(logits, self.bitmask)
                token = int(logits.argmax(dim=-1))
                if matcher is not None:
                    matcher.accept_token(token)
                    finished = matcher.is_terminated()
                else:
                    finished = token in self.stop_ids
                if finished:
                    break
                generated.append(token)
                input_ids = torch.tensor([[token]])
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "finished": finished,
            "new_tokens": len(generated),
            "seconds": round(time.monotonic() - started, 2),
        }


# --------------------------------------------------------------------------- sample


def load_test(schema_id: str, test_index: int) -> tuple[dict, Any]:
    payload = json.loads((TESTS_DIR / schema_id).read_text(encoding="utf-8"))
    return payload["schema"], payload["tests"][test_index]["data"]


def dump_instance(instance: Any) -> str:
    # Same serialization as run_per_test_framework_logging.py.
    return json.dumps(instance, indent=None, ensure_ascii=False)


def held_out_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    """Test split of the rule-extraction script (grouped by schema_id)."""
    sys.path.insert(0, str(EXT_ROOT / "scripts"))
    from extract_coverage_decision_tree_rules import split_rows  # noqa: WPS433

    return split_rows(rows, seed, 0.15, 0.15)["test"]


def cmd_sample(args: argparse.Namespace) -> None:
    engine = Engine(args.model, load_model=False)
    rng = random.Random(args.seed)
    cases = []
    for target in ("over", "under"):
        rules = load_rules(target)
        with open(FEATURE_TABLES[target], encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["expected_validity"] == EXPECTED_VALIDITY[target]]
        if target == "under":
            rows = held_out_rows(rows, args.rules_seed)
        rng.shuffle(rows)
        quota = {True: args.per_group, False: args.per_group}
        used_schemas: set[str] = set()
        for row in rows:
            if not any(quota.values()):
                break
            fired = fired_rules(row, rules)
            flagged = bool(fired)
            if not quota[flagged] or row["schema_id"] in used_schemas:
                continue
            schema, instance = load_test(row["schema_id"], int(row["test_index"]))
            if not isinstance(instance, dict):
                continue
            target_text = dump_instance(instance)
            # Guard against rows whose test_index does not point to the instance that was run.
            if evaluate(target_text, schema, instance)["schema_valid"] != (row["expected_validity"] == "valid"):
                continue
            target_tokens = len(engine.tokenizer.encode(target_text, add_special_tokens=False))
            schema_chars = len(json.dumps(schema))
            if target_tokens > args.max_target_tokens or schema_chars > args.max_schema_chars[target]:
                continue
            try:
                grammar = engine.compile(schema)
            except Exception:  # schema not supported by XGrammar: no decision to compare with
                continue
            used_schemas.add(row["schema_id"])
            quota[flagged] -= 1
            cases.append(
                {
                    "case_id": f"{target}::{row['test_id']}",
                    "target": target,
                    "group": "flagged" if flagged else "control",
                    "fired_rules": fired,
                    "dataset": row["dataset"],
                    "schema_id": row["schema_id"],
                    "test_index": int(row["test_index"]),
                    "expected_validity": row["expected_validity"],
                    # Label measured during the internship (Llama 3.1 tokenizer).
                    "internship_failure_type": row["failure_type"],
                    # Same check redone here with this model's tokenizer.
                    "mask_accepts_target": engine.mask_accepts(grammar, target_text),
                    "target_tokens": target_tokens,
                    "schema_chars": schema_chars,
                }
            )
            print(f"[sample] {cases[-1]['case_id']} group={cases[-1]['group']} tokens={target_tokens}", flush=True)
    with open(args.sample, "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} cases to {args.sample}")


# --------------------------------------------------------------------------- run


def copy_prompt(target_text: str) -> str:
    return f"Return the following record as a JSON object, keeping every key and value unchanged.\n\nRecord:\n{target_text}"


def schema_prompt(target_text: str, schema: dict) -> str:
    return (
        "Return the following record as a JSON object that is valid against the JSON Schema below. "
        "Keep the record unchanged wherever it already respects the schema, and correct only what violates it "
        "(wrong types, values outside the allowed bounds or enums, missing required properties, forbidden properties).\n\n"
        f"JSON Schema:\n{json.dumps(schema)}\n\nRecord:\n{target_text}"
    )


def evaluate(text: str, schema: dict, instance: Any) -> dict[str, Any]:
    import jsonschema

    # Unconstrained answers are often wrapped in a markdown code fence.
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    try:
        parsed = json.loads(fenced.group(1) if fenced else text)
    except ValueError:
        return {"parsed": False, "schema_valid": False, "equals_target": False, "validation_error": "output is not valid JSON"}
    validator_class = jsonschema.validators.validator_for(schema)
    # The benchmark treats `format` as an assertion.
    validator = validator_class(schema, format_checker=validator_class.FORMAT_CHECKER)
    error = next(iter(validator.iter_errors(parsed)), None)
    return {
        "parsed": True,
        "schema_valid": error is None,
        "equals_target": parsed == instance,
        "validation_error": "" if error is None else error.message[:300],
    }


def attempt(engine: Engine, prompt: str, grammar: Any | None, max_new_tokens: int, schema: dict, instance: Any) -> dict[str, Any]:
    generation = engine.generate(prompt, grammar, max_new_tokens)
    return {**generation, **evaluate(generation["text"], schema, instance)}


def mitigate(engine: Engine, case: dict, grammar: Any, schema: dict, instance: Any, max_new_tokens: int) -> dict[str, Any]:
    """Rule-guided mitigation, applied to flagged cases only.

    OVER risk: the grammar may forbid a valid output, so generate without it.
    UNDER risk: the grammar may let an invalid output through, so state the schema in the prompt.
    In both cases the output is validated with `jsonschema`, with one retry on failure.
    """
    target_text = dump_instance(instance)
    if case["target"] == "over":
        # Free-form answers are pretty-printed, so they need a larger token budget.
        prompt, used_grammar, max_new_tokens = copy_prompt(target_text), None, 3 * max_new_tokens
    else:
        prompt, used_grammar = schema_prompt(target_text, schema), grammar
    attempts = [attempt(engine, prompt, used_grammar, max_new_tokens, schema, instance)]
    if not attempts[0]["schema_valid"]:
        retry_prompt = (
            f"{prompt}\n\nA previous answer was rejected by the validator: "
            f"{attempts[0]['validation_error']}\nFix this problem."
        )
        attempts.append(attempt(engine, retry_prompt, used_grammar, max_new_tokens, schema, instance))
    return {"attempts": attempts, "final": attempts[-1], "caught_invalid": not attempts[-1]["schema_valid"]}


def cmd_run(args: argparse.Namespace) -> None:
    cases = [json.loads(line) for line in open(args.sample, encoding="utf-8")]
    done = set()
    if args.results.exists():
        done = {json.loads(line)["case_id"] for line in open(args.results, encoding="utf-8")}
    todo = [case for case in cases if case["case_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    engine = Engine(args.model)
    for position, case in enumerate(todo, start=1):
        schema, instance = load_test(case["schema_id"], case["test_index"])
        grammar = engine.compile(schema)
        max_new_tokens = int(case["target_tokens"] * 1.5) + 30
        record = dict(case)
        record["model"] = args.model
        record["baseline"] = attempt(engine, copy_prompt(dump_instance(instance)), grammar, max_new_tokens, schema, instance)
        if case["group"] == "flagged":
            record["mitigation"] = mitigate(engine, case, grammar, schema, instance, max_new_tokens)
        with open(args.results, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        baseline = record["baseline"]
        print(
            f"[run {position}/{len(todo)}] {case['case_id']} group={case['group']} "
            f"valid={baseline['schema_valid']} equals_target={baseline['equals_target']} {baseline['seconds']}s",
            flush=True,
        )


# --------------------------------------------------------------------------- report


def is_error(target: str, outcome: dict[str, Any]) -> bool:
    """OVER: the valid target could not be produced. UNDER: a schema-invalid output was produced."""
    if target == "over":
        return not (outcome["equals_target"] and outcome["schema_valid"])
    return not outcome["schema_valid"]


def cmd_report(args: argparse.Namespace) -> None:
    records = [json.loads(line) for line in open(args.results, encoding="utf-8")]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["target"], record["group"])].append(record)
    rows = []
    for (target, group), items in sorted(groups.items()):
        n = len(items)
        row = {
            "target": target.upper(),
            "group": group,
            "n": n,
            "mask_level_errors": sum(
                item["mask_accepts_target"] == (target == "under") for item in items
            ),
            "baseline_errors": sum(is_error(target, item["baseline"]) for item in items),
            "baseline_unfinished": sum(not item["baseline"]["finished"] for item in items),
            "mitigated_errors": "",
            "mitigated_retries": "",
        }
        if group == "flagged":
            row["mitigated_errors"] = sum(is_error(target, item["mitigation"]["final"]) for item in items)
            row["mitigated_retries"] = sum(len(item["mitigation"]["attempts"]) > 1 for item in items)
        rows.append(row)
    with open(OUT_DIR / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(str(value) for value in row.values()) + " |" for row in rows]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample", "run", "report"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model ID.")
    parser.add_argument("--sample", type=Path, default=OUT_DIR / "sample.jsonl")
    parser.add_argument("--results", type=Path, default=OUT_DIR / "results.jsonl")
    parser.add_argument("--per-group", type=int, default=15, help="Cases per (error type, flagged/control) group.")
    parser.add_argument("--max-target-tokens", type=int, default=120)
    parser.add_argument("--max-over-schema-chars", type=int, default=30000)
    parser.add_argument("--max-under-schema-chars", type=int, default=2000, help="The schema is put in the mitigation prompt.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rules-seed", type=int, default=20260729, help="Seed used by extract_coverage_decision_tree_rules.py.")
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many remaining cases.")
    args = parser.parse_args()
    args.max_schema_chars = {"over": args.max_over_schema_chars, "under": args.max_under_schema_chars}
    {"sample": cmd_sample, "run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
