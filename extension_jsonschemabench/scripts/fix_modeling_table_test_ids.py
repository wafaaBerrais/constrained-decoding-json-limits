#!/usr/bin/env python3
"""Repair wrong `test_id` / `test_index` values in the modeling tables.

In `modeles_predictifs/<framework>/modeling/{under,over}_dataset.csv`, a few
rows carry the number of another test of the same schema. The rest of the row
(expected validity, framework decision, features) is right, so models and
metrics are not affected, but anything that reloads the instance from
`maskbench/data/<schema_id>#/tests/<test_index>` gets the wrong instance.

The raw per-test logs (`per_test_results.jsonl`) have the right numbers. Inside
a schema, the rows of a table are in the same order as the tests in the log,
so each table is re-aligned with the log: the k-th row of a schema is the k-th
logged test with that expected validity. A schema is only rewritten when the
framework decision and the instance features of every row agree with the
aligned test.

Dry run by default; pass --apply to rewrite the CSV files in place.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

csv.field_size_limit(1024 * 1024 * 1024)

ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = ROOT / "extension_jsonschemabench"
MODELS_ROOT = EXT_ROOT / "coverage_prediction" / "modeles_predictifs"
EXPECTED_VALIDITY = {"under": "invalid", "over": "valid"}
RUNTIME_RESULTS = {"passed", "failed"}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_outlines_coverage_prediction import analyze_instance, analyze_schema, collect_string_patterns  # noqa: E402


def load_log(results_root: Path, framework: str) -> dict[str, dict[int, dict[str, Any]]]:
    """All logged GitHub tests of a framework: schema_id -> test_index -> record."""
    log: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in sorted((results_root / framework).glob("Github_*/per_test_results.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    log[str(record["schema_id"])][int(record["test_index"])] = record
    return log


@lru_cache(maxsize=None)
def load_schema_file(schema_id: str) -> tuple[dict, list, dict, Any]:
    payload = json.loads((ROOT / "maskbench" / "data" / schema_id).read_text(encoding="utf-8"))
    schema = payload["schema"]
    return schema, payload["tests"], analyze_schema(schema), collect_string_patterns(schema)


def same_value(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-6
    except (TypeError, ValueError):
        return str(left) == str(right)


def row_matches_test(row: dict[str, str], record: dict[str, Any]) -> bool:
    """Same framework decision and same instance features."""
    if row["outlines_result"] != ("accepted" if record.get("accepted") else "rejected"):
        return False
    schema, tests, schema_features, string_patterns = load_schema_file(row["schema_id"])
    features = analyze_instance(schema, tests[int(record["test_index"])]["data"], schema_features, string_patterns)
    return all(same_value(row[name], value) for name, value in features.items() if name in row and not name.endswith("_bucket"))


def align_schema(rows: list[dict[str, str]], logged: dict[int, dict[str, Any]], validity: str) -> list[int] | None:
    """Test index of each row, or None when the table cannot be aligned with the log."""
    tests = [logged[index] for index in sorted(logged) if logged[index].get("expected_validity") == validity]
    if len(tests) != len(rows):
        tests = [record for record in tests if record.get("actual_result") in RUNTIME_RESULTS]
    if len(tests) != len(rows):
        return None
    if not all(row_matches_test(row, record) for row, record in zip(rows, tests)):
        return None
    return [int(record["test_index"]) for record in tests]


def fix_table(path: Path, log: dict[str, dict[int, dict[str, Any]]], validity: str, apply: bool) -> Counter:
    stats: Counter = Counter()
    with open(path, encoding="utf-8", newline="") as handle:
        text = handle.read()
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(newline)
    rows = list(csv.DictReader(lines))
    if len(rows) != len([line for line in lines[1:] if line]):
        raise ValueError(f"{path}: rows span several lines, cannot patch line by line")
    header = lines[0].split(",")
    if header[:4] != ["dataset", "schema_id", "test_id", "test_index"]:
        raise ValueError(f"{path}: unexpected leading columns {header[:4]}")

    by_schema: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        by_schema[row["schema_id"]].append(position)

    for schema_id, positions in by_schema.items():
        schema_rows = [rows[position] for position in positions]
        aligned = align_schema(schema_rows, log.get(schema_id, {}), validity)
        if aligned is None:
            stats["schemas_not_aligned"] += 1
            stats["rows_left_unchanged_in_those_schemas"] += len(positions)
            wrong = sum(
                log.get(schema_id, {}).get(int(row["test_index"]), {}).get("expected_validity") != validity for row in schema_rows
            )
            stats["rows_still_wrong"] += wrong
            continue
        for position, test_index in zip(positions, aligned):
            stats["rows_checked"] += 1
            if int(rows[position]["test_index"]) == test_index:
                continue
            stats["rows_fixed"] += 1
            dataset, schema, _, _, rest = lines[position + 1].split(",", 4)
            lines[position + 1] = ",".join([dataset, schema, f"{schema_id}::test_{test_index:05d}", str(test_index), rest])
        stats["schemas_fixed"] += any(int(rows[p]["test_index"]) != i for p, i in zip(positions, aligned))

    if apply and stats["rows_fixed"]:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(newline.join(lines))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=EXT_ROOT / "results" / "per_dataset_runs", help="Folder holding <framework>/<dataset>/per_test_results.jsonl.")
    parser.add_argument("--frameworks", nargs="+", default=["xgr", "outlines", "guidance"])
    parser.add_argument("--apply", action="store_true", help="Rewrite the CSV files (default: dry run).")
    args = parser.parse_args()

    for framework in args.frameworks:
        log = load_log(args.results_root, framework)
        if not log:
            print(f"{framework}: no per_test_results.jsonl under {args.results_root / framework}, skipped")
            continue
        for target, validity in EXPECTED_VALIDITY.items():
            path = MODELS_ROOT / framework / "modeling" / f"{target}_dataset.csv"
            if not path.exists():
                continue
            stats = fix_table(path, log, validity, args.apply)
            print(f"{framework} {target}: {dict(stats)}{'' if args.apply else ' (dry run)'}")


if __name__ == "__main__":
    main()
