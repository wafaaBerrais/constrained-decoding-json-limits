#!/usr/bin/env python3
"""Train shallow decision trees and export readable coverage rules."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from build_outlines_coverage_prediction import (
    CATEGORICAL_FEATURES,
    ROOT,
    choose_threshold,
    make_feature_matrix,
    sklearn_confusion_row,
    sklearn_metrics_row,
    split_groups,
    transformed_feature_names,
    write_csv,
)


DEFAULT_COVERAGE_ROOT = ROOT / "extension_jsonschemabench" / "coverage_prediction"
DEFAULT_FRAMEWORKS = ["guidance", "outlines", "xgr"]
DEFAULT_TARGETS = ["under", "over"]


FEATURE_DESCRIPTIONS = {
    "additionalProperties_mode_false": "indicateur binaire : le schema utilise surtout `additionalProperties: false`, donc les proprietes non declarees sont interdites.",
    "additionalProperties_mode_true": "indicateur binaire : le schema utilise surtout `additionalProperties: true`, donc les proprietes non declarees peuvent etre acceptees.",
    "allOf_branch_count": "nombre total de branches dans les blocs `allOf` du schema.",
    "allOf_count": "nombre d'occurrences du mot-cle `allOf` dans le schema.",
    "allOf_satisfied_branch_ratio": "proportion de branches `allOf` satisfaites par l'instance testee.",
    "branches_have_properties": "indicateur binaire : au moins une branche de combinator contient des `properties`.",
    "branches_have_required": "indicateur binaire : au moins une branche de combinator contient des champs `required`.",
    "branches_have_same_type": "indicateur binaire : les branches d'un combinator portent sur le meme type JSON.",
    "combinator_branch_count_avg": "nombre moyen de branches dans les combinators (`allOf`, `anyOf`, `oneOf`).",
    "combinator_branch_count_min": "nombre minimum de branches observe dans les combinators du schema.",
    "combinator_count": "nombre total de combinators (`allOf`, `anyOf`, `oneOf`) dans le schema.",
    "combinator_depth_max": "profondeur maximale d'un combinator dans le schema.",
    "combinator_type_oneOf": "indicateur binaire : le combinator dominant est `oneOf`.",
    "complex_keywords_same_node_avg": "nombre moyen de mots-cles complexes presents sur un meme noeud du schema.",
    "logical_complex_keywords_same_node_count": "nombre maximal de mots-cles logiques complexes presents sur un meme noeud du schema.",
    "object_complex_keywords_same_node_count": "nombre maximal de mots-cles objet complexes presents sur un meme noeud du schema.",
    "regex_complex_keywords_same_node_count": "nombre maximal de mots-cles regex/string complexes presents sur un meme noeud du schema.",
    "enum_context_count": "nombre de contraintes `enum` dans le schema.",
    "enum_validation_case_enum_mismatch": "indicateur binaire : l'instance contient une valeur qui ne correspond pas a l'enum attendu.",
    "instance_matching_pattern_keys_count": "nombre de cles de l'instance qui matchent une contrainte `patternProperties`.",
    "instance_max_abs_numeric_value": "plus grande valeur numerique absolue presente dans l'instance testee.",
    "instance_numeric_value_count": "nombre de valeurs numeriques presentes dans l'instance testee, entiers et flottants, booleens exclus.",
    "instance_max_container_depth": "profondeur maximale des conteneurs dans l'instance testee, en comptant objets et tableaux.",
    "instance_num_properties": "nombre de proprietes au premier niveau de l'objet instance.",
    "instance_string_max_length": "longueur de la plus grande chaine de caracteres dans l'instance testee.",
    "instance_string_total_length": "somme des longueurs de toutes les chaines de caracteres dans l'instance testee.",
    "numeric_boundary_case_equal_min": "indicateur binaire : la valeur numerique testee est exactement sur la borne minimale.",
    "numeric_boundary_case_inside_range": "indicateur binaire : la valeur numerique testee est a l'interieur de l'intervalle autorise.",
    "numeric_boundary_case_not_applicable": "indicateur binaire : aucun cas de borne numerique pertinent n'a ete detecte pour cette instance.",
    "numeric_depth": "profondeur maximale ou moyenne des contraintes numeriques dans le schema.",
    "object_additionalProperties_value_true": "indicateur binaire : au moins un contexte objet utilise `additionalProperties: true`.",
    "object_additional_properties_case_extra_allowed_or_absent": "indicateur binaire : l'instance a des proprietes extra mais le schema les autorise, ou bien il n'y a pas de restriction claire.",
    "object_additional_properties_case_no_extra_properties": "indicateur binaire : l'instance n'a pas de proprietes extra par rapport aux `properties` du schema.",
    "object_context_count": "nombre de contextes objet detectes dans le schema.",
    "object_properties_count_max": "nombre maximal de proprietes declarees dans un meme objet du schema.",
    "object_required_count_bucket_6+": "indicateur binaire : le nombre maximal de champs `required` est dans le bucket `6+`.",
    "object_required_count_max": "nombre maximal de champs `required` dans un meme objet du schema.",
    "object_required_subset_of_properties": "indicateur binaire historique : au moins un contexte objet a ses champs `required` inclus dans `properties`.",
    "object_required_subset_valid_context_count": "nombre de contextes objet ou `required` est non vide et entierement inclus dans `properties`.",
    "object_required_subset_invalid_context_count": "nombre de contextes objet ou au moins un champ `required` est absent de `properties`.",
    "object_has_required_outside_properties": "indicateur binaire : au moins un contexte objet declare un champ `required` absent de `properties`.",
    "patternProperties_pattern_count": "nombre de patterns declares dans `patternProperties`.",
    "patternProperties_with_properties_count": "nombre de contextes ou `patternProperties` et `properties` apparaissent ensemble.",
    "schema_dependentSchemas_count": "nombre d'occurrences du mot-cle `dependentSchemas`.",
    "schema_dependentRequired_count": "nombre d'occurrences du mot-cle `dependentRequired`.",
    "schema_legacy_dependencies_count": "nombre d'occurrences du mot-cle historique `dependencies`.",
    "same_node_properties_and_additionalProperties": "indicateur binaire : un meme noeud du schema contient a la fois `properties` et `additionalProperties`.",
    "schema_total_properties_recursive": "nombre total de proprietes declarees dans tout le schema, en comptant recursivement les sous-objets.",
    "string_context_count": "nombre de contextes string dans le schema, par exemple `type: string`, `pattern`, `minLength`, `maxLength` ou `format`.",
    "string_length_violation_count_bucket_0": "indicateur binaire : aucune violation de longueur de string n'a ete detectee.",
    "string_pattern_has_alternation": "indicateur binaire : au moins une regex string contient une alternation, par exemple `a|b`.",
    "string_pattern_violation_count": "nombre de valeurs string de l'instance qui violent une contrainte `pattern`.",
    "any_visited_type_mismatch": "indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.",
    "unmatched_keys_allowed_by_additionalProperties": "nombre de cles de l'instance non declarees dans `properties` mais acceptees via `additionalProperties`.",
    "unmatched_keys_allowed_by_additionalProperties_bucket_2": "indicateur binaire : le nombre de cles non declarees mais acceptees tombe dans le bucket `2`.",
}

SCHEMA_FEATURES = {
    "additionalProperties_mode_false",
    "additionalProperties_mode_true",
    "allOf_branch_count",
    "allOf_count",
    "branches_have_properties",
    "branches_have_required",
    "branches_have_same_type",
    "combinator_branch_count_avg",
    "combinator_branch_count_min",
    "combinator_count",
    "combinator_depth_max",
    "combinator_type_oneOf",
    "complex_keywords_same_node_avg",
    "logical_complex_keywords_same_node_count",
    "object_complex_keywords_same_node_count",
    "regex_complex_keywords_same_node_count",
    "enum_context_count",
    "numeric_depth",
    "object_additionalProperties_value_true",
    "object_context_count",
    "object_properties_count_max",
    "object_required_count_bucket_6+",
    "object_required_count_max",
    "object_required_subset_of_properties",
    "object_required_subset_valid_context_count",
    "object_required_subset_invalid_context_count",
    "object_has_required_outside_properties",
    "patternProperties_pattern_count",
    "patternProperties_with_properties_count",
    "schema_dependentSchemas_count",
    "schema_dependentRequired_count",
    "schema_legacy_dependencies_count",
    "same_node_properties_and_additionalProperties",
    "schema_total_properties_recursive",
    "string_context_count",
    "string_pattern_has_alternation",
}

INSTANCE_FEATURES = {
    "allOf_satisfied_branch_ratio",
    "enum_validation_case_enum_mismatch",
    "instance_matching_pattern_keys_count",
    "instance_max_abs_numeric_value",
    "instance_numeric_value_count",
    "instance_max_container_depth",
    "instance_num_properties",
    "instance_string_max_length",
    "instance_string_total_length",
    "numeric_boundary_case_equal_min",
    "numeric_boundary_case_inside_range",
    "numeric_boundary_case_not_applicable",
    "object_additional_properties_case_extra_allowed_or_absent",
    "object_additional_properties_case_no_extra_properties",
    "string_length_violation_count_bucket_0",
    "string_pattern_violation_count",
    "any_visited_type_mismatch",
    "unmatched_keys_allowed_by_additionalProperties",
    "unmatched_keys_allowed_by_additionalProperties_bucket_2",
}

BINARY_CONDITION_LABELS = {
    "additionalProperties_mode_false": (
        "le schema utilise principalement `additionalProperties: false`",
        "le schema n'utilise pas principalement `additionalProperties: false`",
    ),
    "additionalProperties_mode_true": (
        "le schema utilise principalement `additionalProperties: true`",
        "le schema n'utilise pas principalement `additionalProperties: true`",
    ),
    "branches_have_properties": (
        "des branches de combinator declarent des `properties`",
        "les branches de combinator ne declarent pas de `properties`",
    ),
    "branches_have_required": (
        "des branches de combinator declarent des champs `required`",
        "les branches de combinator ne declarent pas de champs `required`",
    ),
    "branches_have_same_type": (
        "les branches du combinator portent sur le meme type JSON",
        "les branches du combinator ne portent pas toutes sur le meme type JSON",
    ),
    "combinator_type_oneOf": (
        "le schema utilise surtout `oneOf` comme combinator",
        "le schema n'utilise pas surtout `oneOf` comme combinator",
    ),
    "enum_validation_case_enum_mismatch": (
        "l'instance contient une valeur qui ne correspond pas a l'enum attendu",
        "l'instance ne contient pas de mismatch enum detecte",
    ),
    "numeric_boundary_case_equal_min": (
        "la valeur numerique de l'instance est exactement sur la borne minimale",
        "la valeur numerique de l'instance n'est pas exactement sur la borne minimale",
    ),
    "numeric_boundary_case_inside_range": (
        "la valeur numerique de l'instance est dans l'intervalle autorise",
        "la valeur numerique de l'instance n'est pas dans un cas inside_range",
    ),
    "numeric_boundary_case_not_applicable": (
        "il n'y a pas de cas de borne numerique applicable",
        "un cas de borne numerique est applicable",
    ),
    "object_additionalProperties_value_true": (
        "au moins un objet du schema autorise les proprietes extra avec `additionalProperties: true`",
        "aucun objet important du schema n'autorise explicitement les proprietes extra avec `additionalProperties: true`",
    ),
    "object_additional_properties_case_extra_allowed_or_absent": (
        "l'instance a des proprietes extra autorisees ou sans restriction claire",
        "l'instance n'est pas dans le cas extra autorise/absent",
    ),
    "object_additional_properties_case_no_extra_properties": (
        "l'instance n'a pas de proprietes extra par rapport au schema",
        "l'instance a au moins une propriete extra ou un cas different",
    ),
    "object_required_count_bucket_6+": (
        "un objet du schema a 6 champs `required` ou plus",
        "aucun objet du schema n'a 6 champs `required` ou plus",
    ),
    "same_node_properties_and_additionalProperties": (
        "un meme noeud du schema combine `properties` et `additionalProperties`",
        "aucun noeud important ne combine `properties` et `additionalProperties`",
    ),
    "string_length_violation_count_bucket_0": (
        "aucune violation de longueur string n'est detectee dans l'instance",
        "au moins une violation de longueur string est detectee dans l'instance",
    ),
    "string_pattern_has_alternation": (
        "au moins une regex string contient une alternation de type `a|b`",
        "les regex string ne contiennent pas d'alternation detectee",
    ),
    "any_visited_type_mismatch": (
        "au moins une verification `type` visitee ne correspond pas a l'instance",
        "aucune verification `type` visitee ne mismatch",
    ),
    "unmatched_keys_allowed_by_additionalProperties_bucket_2": (
        "exactement le bucket `2` de cles non declarees mais autorisees est detecte",
        "on n'est pas dans le bucket `2` de cles non declarees mais autorisees",
    ),
}

COUNT_CONDITION_LABELS = {
    "allOf_branch_count": "le schema a {value} branches `allOf` au total",
    "allOf_count": "le schema contient {value} blocs `allOf`",
    "allOf_satisfied_branch_ratio": "l'instance satisfait une proportion de branches `allOf` {value}",
    "combinator_branch_count_avg": "les combinators du schema ont en moyenne {value} branches",
    "combinator_branch_count_min": "les combinators du schema ont au minimum {value} branches",
    "combinator_count": "le schema contient {value} combinators (`allOf`, `anyOf`, `oneOf`)",
    "combinator_depth_max": "le schema a des combinators a une profondeur {value}",
    "complex_keywords_same_node_avg": "les noeuds du schema combinent en moyenne {value} mots-cles complexes",
    "logical_complex_keywords_same_node_count": "un noeud du schema combine {value} mots-cles logiques complexes",
    "object_complex_keywords_same_node_count": "un noeud du schema combine {value} mots-cles objet complexes",
    "regex_complex_keywords_same_node_count": "un noeud du schema combine {value} mots-cles regex/string complexes",
    "enum_context_count": "le schema contient {value} contraintes `enum`",
    "instance_matching_pattern_keys_count": "l'instance contient {value} cles qui matchent `patternProperties`",
    "instance_max_abs_numeric_value": "l'instance contient une valeur numerique absolue {value}",
    "instance_numeric_value_count": "l'instance contient {value} valeurs numeriques",
    "instance_max_container_depth": "l'instance a une profondeur de conteneurs {value}",
    "instance_num_properties": "l'instance contient {value} proprietes au premier niveau",
    "instance_string_max_length": "la plus longue chaine de l'instance a une longueur {value}",
    "instance_string_total_length": "la longueur totale des chaines de l'instance est {value}",
    "numeric_depth": "les contraintes numeriques du schema apparaissent a une profondeur {value}",
    "object_context_count": "le schema contient {value} contextes objet",
    "object_properties_count_max": "un objet du schema declare {value} proprietes",
    "object_required_count_max": "un objet du schema declare {value} champs `required`",
    "object_required_subset_valid_context_count": "le schema contient {value} contextes objet ou `required` est inclus dans `properties`",
    "object_required_subset_invalid_context_count": "le schema contient {value} contextes objet avec un champ `required` absent de `properties`",
    "patternProperties_pattern_count": "le schema declare {value} patterns dans `patternProperties`",
    "patternProperties_with_properties_count": "le schema contient {value} contextes combinant `patternProperties` et `properties`",
    "schema_dependentSchemas_count": "le schema contient {value} occurrences de `dependentSchemas`",
    "schema_dependentRequired_count": "le schema contient {value} occurrences de `dependentRequired`",
    "schema_legacy_dependencies_count": "le schema contient {value} occurrences de `dependencies`",
    "schema_total_properties_recursive": "le schema declare {value} proprietes au total en comptant les sous-objets",
    "string_context_count": "le schema contient {value} contextes string",
    "string_pattern_violation_count": "l'instance contient {value} violations de `pattern` string",
    "unmatched_keys_allowed_by_additionalProperties": "l'instance contient {value} cles non declarees mais autorisees par `additionalProperties`",
}

FEATURE_DESCRIPTIONS.update(
    {
        "branches_have_different_type": "indicateur binaire : au moins un combinator a des branches heterogenes en type declare.",
        "combinator_same_type_count": "nombre de combinators dont les branches sont homogenes en type declare.",
        "combinator_different_type_count": "nombre de combinators dont les branches sont heterogenes en type declare.",
        "combinator_same_type_ratio": "proportion de combinators homogenes parmi les combinators observes.",
        "combinator_different_type_ratio": "proportion de combinators heterogenes parmi les combinators observes.",
        "combinator_have_required_count": "nombre de combinators ayant au moins une branche contenant `required`.",
        "branches_with_required_count": "nombre total de branches de combinator contenant `required`.",
        "branches_with_required_ratio": "proportion de branches de combinator contenant `required`.",
        "combinator_with_additionalProperties_true": "indicateur binaire : une branche de combinator contient `additionalProperties: true`.",
        "combinator_with_additionalProperties_false": "indicateur binaire : une branche de combinator contient `additionalProperties: false`.",
        "combinator_with_additionalProperties_schema": "indicateur binaire : une branche de combinator contient `additionalProperties` sous forme de schema.",
        "allOf_is_satisfied_for_instance": "indicateur binaire : toutes les occurrences `allOf` observees sont satisfaites par l'instance.",
        "anyOf_is_satisfied_for_instance": "indicateur binaire : toutes les occurrences `anyOf` observees satisfont au moins une branche.",
        "oneOf_is_satisfied_for_instance": "indicateur binaire : toutes les occurrences `oneOf` observees satisfont exactement une branche.",
        "allOf_satisfied_all_branch_count": "nombre d'occurrences `allOf` dont toutes les branches sont satisfaites.",
        "allOf_ratio_avg": "moyenne des ratios branches satisfaites / branches totales, calculee occurrence par occurrence pour `allOf`.",
        "anyOf_ratio_avg": "moyenne des ratios branches satisfaites / branches totales, calculee occurrence par occurrence pour `anyOf`.",
        "oneOf_ratio_avg": "moyenne des ratios branches satisfaites / branches totales, calculee occurrence par occurrence pour `oneOf`.",
        "combinator_depth_max": (
            "profondeur maximale selon la convention interne du script : +2 via `properties`, "
            "+2 via une branche de combinator, +1 via `not`, +1 via `items`."
        ),
    }
)

SCHEMA_FEATURES.update(
    {
        "branches_have_different_type",
        "combinator_same_type_count",
        "combinator_different_type_count",
        "combinator_same_type_ratio",
        "combinator_different_type_ratio",
        "combinator_have_required_count",
        "branches_with_required_count",
        "branches_with_required_ratio",
        "combinator_with_additionalProperties_true",
        "combinator_with_additionalProperties_false",
        "combinator_with_additionalProperties_schema",
    }
)

INSTANCE_FEATURES.update(
    {
        "allOf_is_satisfied_for_instance",
        "anyOf_is_satisfied_for_instance",
        "oneOf_is_satisfied_for_instance",
        "allOf_satisfied_all_branch_count",
        "allOf_ratio_avg",
        "anyOf_ratio_avg",
        "oneOf_ratio_avg",
    }
)

BINARY_CONDITION_LABELS.update(
    {
        "branches_have_different_type": (
            "au moins un combinator a des branches de types declares differents",
            "aucun combinator n'a de branches de types declares differents",
        ),
        "combinator_with_additionalProperties_true": (
            "une branche de combinator contient `additionalProperties: true`",
            "aucune branche de combinator ne contient `additionalProperties: true`",
        ),
        "combinator_with_additionalProperties_false": (
            "une branche de combinator contient `additionalProperties: false`",
            "aucune branche de combinator ne contient `additionalProperties: false`",
        ),
        "combinator_with_additionalProperties_schema": (
            "une branche de combinator contient `additionalProperties` sous forme de schema",
            "aucune branche de combinator ne contient `additionalProperties` sous forme de schema",
        ),
    }
)

COUNT_CONDITION_LABELS.update(
    {
        "combinator_same_type_count": "le schema contient {value} combinators homogenes en type declare",
        "combinator_different_type_count": "le schema contient {value} combinators heterogenes en type declare",
        "combinator_same_type_ratio": "la proportion de combinators homogenes est {value}",
        "combinator_different_type_ratio": "la proportion de combinators heterogenes est {value}",
        "combinator_have_required_count": "{value} combinators ont au moins une branche avec `required`",
        "branches_with_required_count": "{value} branches de combinator contiennent `required`",
        "branches_with_required_ratio": "la proportion de branches de combinator avec `required` est {value}",
        "allOf_satisfied_all_branch_count": "{value} occurrences `allOf` ont toutes leurs branches satisfaites",
        "allOf_ratio_avg": "le ratio moyen de branches `allOf` satisfaites est {value}",
        "anyOf_ratio_avg": "le ratio moyen de branches `anyOf` satisfaites est {value}",
        "oneOf_ratio_avg": "le ratio moyen de branches `oneOf` satisfaites est {value}",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-root", default=str(DEFAULT_COVERAGE_ROOT))
    parser.add_argument("--frameworks", nargs="+", default=DEFAULT_FRAMEWORKS)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--max-depths", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--min-samples-leaf", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--refresh-markdown-only", action="store_true", help="Rewrite rule markdown from existing CSV outputs without retraining trees.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_feature_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("pr_auc", 0.0)),
        float(row.get("f1", 0.0)),
        float(row.get("recall", 0.0)),
        float(row.get("precision", 0.0)),
    )


def split_rows(rows: list[dict[str, Any]], seed: int, validation_size: float, test_size: float) -> dict[str, list[dict[str, Any]]]:
    split_schema_ids = split_groups(rows, seed, validation_size, test_size)
    return {
        split: [row for row in rows if str(row["schema_id"]) in schema_ids]
        for split, schema_ids in split_schema_ids.items()
    }


def clean_numeric_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=1e9, neginf=-1e9)
    return np.clip(array, -1e9, 1e9)


def make_rule_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric_indices = [idx for idx, feature in enumerate(features) if feature not in CATEGORICAL_FEATURES]
    categorical_indices = [idx for idx, feature in enumerate(features) if feature in CATEGORICAL_FEATURES]
    transformers = []
    if numeric_indices:
        transformers.append(("numeric", FunctionTransformer(clean_numeric_values, feature_names_out="one-to-one"), numeric_indices))
    if categorical_indices:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_indices))
    return ColumnTransformer(transformers, sparse_threshold=0.0, verbose_feature_names_out=False)


def fit_candidate(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    features: list[str],
    label: str,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
) -> tuple[Pipeline, float, dict[str, Any]]:
    pipeline = Pipeline(
        steps=[
            ("preprocess", make_rule_preprocessor(features)),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    x_train = make_feature_matrix(train_rows, features)
    y_train = np.asarray([int(row[label]) for row in train_rows], dtype=int)
    pipeline.fit(x_train, y_train)

    x_validation = make_feature_matrix(validation_rows, features)
    y_validation = np.asarray([int(row[label]) for row in validation_rows], dtype=int)
    validation_score = pipeline.predict_proba(x_validation)[:, 1]
    threshold = choose_threshold(y_validation, validation_score)
    metrics = sklearn_metrics_row(
        f"tree_depth{max_depth}_leaf{min_samples_leaf}",
        "validation",
        y_validation,
        validation_score,
        threshold=threshold,
    )
    metrics.update({"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "threshold": threshold})
    return pipeline, threshold, metrics


def condition_text(feature: str, threshold: float, go_left: bool) -> str:
    op = "<=" if go_left else ">"
    return f"{feature} {op} {threshold:.6g}"


def parse_rule_conditions(rule: str) -> list[tuple[str, str, float]]:
    conditions: list[tuple[str, str, float]] = []
    if rule == "always":
        return conditions
    for part in rule.split(" AND "):
        match = re.match(r"^([A-Za-z0-9_+]+)\s*(<=|>)\s*([-+0-9.eE]+)$", part.strip())
        if not match:
            continue
        feature, operator, threshold = match.groups()
        conditions.append((feature, operator, float(threshold)))
    return conditions


def is_binary_threshold(feature: str, threshold: float) -> bool:
    if not math.isclose(threshold, 0.5, abs_tol=1e-9):
        return False
    binary_prefixes = (
        "additionalProperties_mode_",
        "numeric_boundary_case_",
        "enum_validation_case_",
        "string_length_violation_count_bucket_",
        "unmatched_keys_allowed_by_additionalProperties_bucket_",
    )
    binary_features = {
        "branches_have_properties",
        "branches_have_required",
        "branches_have_same_type",
        "combinator_type_oneOf",
        "object_additionalProperties_value_true",
        "object_additional_properties_case_extra_allowed_or_absent",
        "object_additional_properties_case_no_extra_properties",
        "same_node_properties_and_additionalProperties",
        "string_pattern_has_alternation",
    }
    return feature in binary_features or feature.startswith(binary_prefixes)


def threshold_phrase(feature: str, operator: str, threshold: float) -> str:
    if is_binary_threshold(feature, threshold):
        return "la condition est active/presente" if operator == ">" else "la condition est absente/non active"
    if threshold.is_integer():
        value = str(int(threshold))
    else:
        value = f"{threshold:.6g}"
    if abs(threshold % 1 - 0.5) < 1e-9:
        if operator == ">":
            return f"valeur superieure a {value}, donc typiquement au moins {math.floor(threshold) + 1}"
        return f"valeur inferieure ou egale a {value}, donc typiquement au plus {math.floor(threshold)}"
    return f"valeur {operator} {value}"


def condition_sentence(feature: str, operator: str, threshold: float) -> str:
    return f"`{feature} {operator} {threshold:.6g}` : {threshold_phrase(feature, operator, threshold)}."


def feature_description(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, "feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.")


def human_value(operator: str, threshold: float) -> str:
    if threshold.is_integer():
        value = str(int(threshold))
    else:
        value = f"{threshold:.6g}"
    if abs(threshold % 1 - 0.5) < 1e-9:
        if operator == ">":
            return f"au moins {math.floor(threshold) + 1}"
        return f"au plus {math.floor(threshold)}"
    if operator == ">":
        return f"superieure a {value}"
    return f"inferieure ou egale a {value}"


def human_condition(feature: str, operator: str, threshold: float) -> str:
    if is_binary_threshold(feature, threshold):
        positive, negative = BINARY_CONDITION_LABELS.get(
            feature,
            (f"`{feature}` est actif", f"`{feature}` est absent"),
        )
        return positive if operator == ">" else negative
    template = COUNT_CONDITION_LABELS.get(feature)
    if template:
        return template.format(value=human_value(operator, threshold))
    return f"`{feature}` est {human_value(operator, threshold)}"


def contextualize_condition(sentence: str, context: str) -> str:
    if context == "schema":
        replacements = (
            ("le schema n'utilise", "ils n'utilisent"),
            ("le schema utilise", "ils utilisent"),
            ("le schema declare", "ils declarent"),
            ("le schema contient", "ils contiennent"),
            ("le schema a", "ils ont"),
            ("un objet du schema declare", "un de leurs objets declare"),
            ("au moins un objet du schema", "au moins un de leurs objets"),
            ("aucun objet du schema", "aucun de leurs objets"),
            ("les noeuds du schema", "leurs noeuds"),
            ("les combinators du schema", "leurs combinators"),
            ("les regex de `patternProperties`", "leurs regex de `patternProperties`"),
        )
    elif context == "instance":
        replacements = (
            ("l'instance ne contient", "elle ne contient"),
            ("l'instance contient", "elle contient"),
            ("l'instance n'a", "elle n'a"),
            ("l'instance a", "elle a"),
            ("l'instance satisfait", "elle satisfait"),
        )
    else:
        replacements = ()
    for old, new in replacements:
        if sentence.startswith(old):
            return new + sentence[len(old) :]
    return sentence


def split_human_conditions(conditions: list[tuple[str, str, float]]) -> tuple[list[str], list[str], list[str]]:
    schema_parts: list[str] = []
    instance_parts: list[str] = []
    other_parts: list[str] = []
    for feature, operator, threshold in conditions:
        sentence = human_condition(feature, operator, threshold)
        if feature in SCHEMA_FEATURES:
            schema_parts.append(contextualize_condition(sentence, "schema"))
        elif feature in INSTANCE_FEATURES:
            instance_parts.append(contextualize_condition(sentence, "instance"))
        else:
            other_parts.append(sentence)
    return schema_parts, instance_parts, other_parts


def join_french(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " et " + items[-1]


def interpretation_text(framework: str, target: str, rule: dict[str, Any]) -> str:
    conditions = parse_rule_conditions(str(rule["rule"]))
    schema_parts, instance_parts, other_parts = split_human_conditions(conditions)
    prediction = (
        "le modele predit **OVER**"
        if target == "over"
        else "le modele predit **UNDER**"
    )
    meaning = (
        "cela veut dire : risque que le framework rejette une instance qui devrait etre acceptee"
        if target == "over"
        else "cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee"
    )
    support = int(rule.get("test_support", 0))
    precision = float(rule.get("test_precision", 0.0))
    blocks = []
    if schema_parts:
        blocks.append(f"Dans les schemas ou {join_french(schema_parts)}")
    if instance_parts:
        prefix = "et pour une instance ou" if schema_parts else "Pour une instance ou"
        blocks.append(f"{prefix} {join_french(instance_parts)}")
    if other_parts:
        prefix = "avec aussi" if blocks else "Quand"
        blocks.append(f"{prefix} {join_french(other_parts)}")
    condition_text_block = ", ".join(blocks) if blocks else "Dans tous les cas"
    return (
        f"{condition_text_block}, alors pour `{framework}` {prediction}. "
        f"Donc {meaning}. "
        f"Dans le test, cette regle couvre {support} cas ; parmi les cas couverts, la precision est {precision:.3f}."
    )


def leaf_rules(
    pipeline: Pipeline,
    features: list[str],
    selected_threshold: float,
    target: str,
) -> list[dict[str, Any]]:
    classifier = pipeline.named_steps["classifier"]
    names = transformed_feature_names(pipeline, features)
    tree = classifier.tree_
    rules: list[dict[str, Any]] = []

    def walk(node_id: int, path: list[str]) -> None:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:
            counts = tree.value[node_id][0]
            negative_value = float(counts[0])
            positive_value = float(counts[1]) if len(counts) > 1 else 0.0
            total_value = negative_value + positive_value
            positive_rate = positive_value / total_value if total_value else 0.0
            support = float(tree.weighted_n_node_samples[node_id])
            positive = positive_rate * support
            negative = (1.0 - positive_rate) * support
            predicted_positive = positive_rate >= selected_threshold
            rules.append(
                {
                    "target": target,
                    "leaf_id": node_id,
                    "predicted_class": target.upper() if predicted_positive else "negative",
                    "positive_rate_train_leaf": positive_rate,
                    "positive_count_train_leaf_weighted": positive,
                    "negative_count_train_leaf_weighted": negative,
                    "weighted_support_train_leaf": support,
                    "n_conditions": len(path),
                    "rule": " AND ".join(path) if path else "always",
                }
            )
            return
        feature_name = names[tree.feature[node_id]]
        threshold = float(tree.threshold[node_id])
        walk(left, path + [condition_text(feature_name, threshold, True)])
        walk(right, path + [condition_text(feature_name, threshold, False)])

    walk(0, [])
    return sorted(
        rules,
        key=lambda row: (
            row["predicted_class"] != target.upper(),
            -float(row["positive_rate_train_leaf"]),
            int(row["n_conditions"]),
        ),
    )


def add_test_rule_metrics(
    rules: list[dict[str, Any]],
    pipeline: Pipeline,
    features: list[str],
    test_rows: list[dict[str, Any]],
    label: str,
) -> None:
    classifier = pipeline.named_steps["classifier"]
    leaf_ids = classifier.apply(pipeline.named_steps["preprocess"].transform(make_feature_matrix(test_rows, features)))
    labels = np.asarray([int(row[label]) for row in test_rows], dtype=int)
    by_leaf: dict[int, list[int]] = {}
    for idx, leaf_id in enumerate(leaf_ids):
        by_leaf.setdefault(int(leaf_id), []).append(idx)
    for rule in rules:
        indices = by_leaf.get(int(rule["leaf_id"]), [])
        positives = int(labels[indices].sum()) if indices else 0
        support = len(indices)
        rule["test_support"] = support
        rule["test_positives"] = positives
        rule["test_precision"] = positives / support if support else 0.0
        rule["test_recall_contribution"] = positives / int(labels.sum()) if int(labels.sum()) else 0.0


def plot_tree_svg(path: Path, pipeline: Pipeline, features: list[str], target: str, title: str, selected_threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classifier = pipeline.named_steps["classifier"]
    names = transformed_feature_names(pipeline, features)
    tree = classifier.tree_
    depth = classifier.get_depth()
    leaf_counter = 0
    positions: dict[int, tuple[float, float]] = {}

    def layout(node_id: int, node_depth: int) -> float:
        nonlocal leaf_counter
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:
            x = 110.0 + leaf_counter * 230.0
            leaf_counter += 1
        else:
            left_x = layout(left, node_depth + 1)
            right_x = layout(right, node_depth + 1)
            x = (left_x + right_x) / 2.0
        positions[node_id] = (x, 80.0 + node_depth * 165.0)
        return x

    layout(0, 0)
    width = max(900.0, 220.0 + max((x for x, _ in positions.values()), default=900.0))
    height = max(300.0, 160.0 + (depth + 1) * 165.0)
    box_w = 190.0
    box_h = 98.0

    def svg_text(x: float, y: float, text: str, size: int = 11, anchor: str = "middle", weight: str = "400") -> str:
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Inter, Arial, sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="#1F2933">'
            f"{html.escape(text)}</text>"
        )

    def wrap(text: str, max_len: int = 28) -> list[str]:
        words = text.replace("_", "_ ").split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_len:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word[:max_len]
        if current:
            lines.append(current)
        return lines[:4]

    def node_label(node_id: int) -> list[str]:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        counts = tree.value[node_id][0]
        negative = float(counts[0])
        positive = float(counts[1]) if len(counts) > 1 else 0.0
        total = negative + positive
        positive_rate = positive / total if total else 0.0
        samples = int(tree.n_node_samples[node_id])
        if left == right:
            cls = target.upper() if positive_rate >= selected_threshold else "negative"
            return [f"leaf: {cls}", f"p={positive_rate:.2f}", f"samples={samples}"]
        condition = f"{names[tree.feature[node_id]]} <= {tree.threshold[node_id]:.3g}"
        return wrap(condition, 30) + [f"p={positive_rate:.2f}", f"samples={samples}"]

    body: list[str] = [
        f'<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text(28, 32, title, 18, "start", "700"),
        svg_text(28, 52, f"Shallow decision tree for {target.upper()} rule extraction.", 11, "start"),
    ]

    for node_id, (x, y) in positions.items():
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:
            continue
        for child_id, edge_label in ((left, "<="), (right, ">")):
            child_x, child_y = positions[child_id]
            body.append(
                f'<line x1="{x:.2f}" y1="{y + box_h / 2:.2f}" x2="{child_x:.2f}" y2="{child_y - box_h / 2:.2f}" '
                f'stroke="#A7B0BA" stroke-width="1.2"/>'
            )
            body.append(svg_text((x + child_x) / 2.0, (y + child_y) / 2.0 - 6, edge_label, 10))

    for node_id, (x, y) in positions.items():
        counts = tree.value[node_id][0]
        positive = float(counts[1]) if len(counts) > 1 else 0.0
        negative = float(counts[0])
        positive_rate = positive / (positive + negative) if positive + negative else 0.0
        fill = "#FFE8CC" if positive_rate >= selected_threshold else "#E7F0FF"
        stroke = "#D96C06" if positive_rate >= selected_threshold else "#2F6BFF"
        body.append(
            f'<rect x="{x - box_w / 2:.2f}" y="{y - box_h / 2:.2f}" width="{box_w}" height="{box_h}" '
            f'rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        for idx, line in enumerate(node_label(node_id)):
            body.append(svg_text(x, y - 27 + idx * 17, line, 10 if idx else 11, "middle", "700" if idx == 0 else "400"))

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{html.escape(title)}">'
        + "".join(body)
        + "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def write_rules_markdown(
    path: Path,
    framework: str,
    target: str,
    selected_model: str,
    threshold: float,
    metrics: dict[str, Any],
    rules: list[dict[str, Any]],
    tree_text: str,
) -> None:
    positive_rules = [row for row in rules if row["predicted_class"] == target.upper()]
    positive_rules = sorted(
        positive_rules,
        key=lambda row: (float(row["test_precision"]), int(row["test_support"]), float(row["positive_rate_train_leaf"])),
        reverse=True,
    )
    lines = [
        f"# Regles par arbre de decision - {framework} {target.upper()}",
        "",
        "## Processus",
        "",
        "Cet arbre est un modele interpretable entraine sur la meme table de modelisation et les memes features retenues que le modele principal.",
        "Il est volontairement peu profond : chaque feuille positive peut donc etre lue comme une regle explicite.",
        "",
        "Etapes :",
        "",
        "1. Garder les lignes runtime deja presentes dans la table de modelisation.",
        "2. Utiliser la liste de features retenue dans `modeling/selected_<target>_features.txt`.",
        "3. Faire le split par `schema_id` en train, validation et test, comme dans le pipeline principal.",
        "4. Entrainer plusieurs candidats `DecisionTreeClassifier`.",
        "5. Choisir l'arbre avec PR-AUC validation, puis F1, recall et precision en cas d'egalite.",
        "6. Choisir le seuil de decision sur validation pour maximiser le F1.",
        "7. Exporter chaque feuille positive comme regle candidate.",
        "",
        "## Arbre Retenu",
        "",
        f"- arbre retenu : `{selected_model}`",
        f"- seuil : `{threshold:.4g}`",
        f"- test F1: `{float(metrics['f1']):.6g}`",
        f"- test precision : `{float(metrics['precision']):.6g}`",
        f"- test recall: `{float(metrics['recall']):.6g}`",
        f"- test PR-AUC: `{float(metrics['pr_auc']):.6g}`",
        f"- test ROC-AUC: `{float(metrics['roc_auc']):.6g}`",
        "",
        "## Regles Positives",
        "",
    ]
    if not positive_rules:
        lines.append("Aucune feuille positive n'a ete selectionnee avec le seuil choisi.")
    for idx, rule in enumerate(positive_rules[:20], start=1):
        conditions = parse_rule_conditions(str(rule["rule"]))
        used_features = []
        for feature, _, _ in conditions:
            if feature not in used_features:
                used_features.append(feature)
        lines.extend(
            [
                f"### Regle {idx}",
                "",
                f"- taux positif feuille train : `{float(rule['positive_rate_train_leaf']):.3f}`",
                f"- support test : `{int(rule['test_support'])}`",
                f"- precision test : `{float(rule['test_precision']):.3f}`",
                f"- contribution au recall test : `{float(rule['test_recall_contribution']):.3f}`",
                "",
                "```text",
                str(rule["rule"]),
                "```",
                "",
                "**Features utilisees**",
                "",
            ]
        )
        if used_features:
            for feature in used_features:
                lines.append(f"- `{feature}` : {feature_description(feature)}")
        else:
            lines.append("- Aucune condition explicite : la feuille correspond a la regle globale.")
        lines.extend(
            [
                "",
                "**Seuils de la regle**",
                "",
            ]
        )
        if conditions:
            for feature, operator, threshold in conditions:
                lines.append(f"- {condition_sentence(feature, operator, threshold)}")
        else:
            lines.append("- Pas de seuil : prediction constante.")
        lines.extend(
            [
                "",
                "**Interpretation en francais**",
                "",
                interpretation_text(framework, target, rule),
                "",
            ]
        )
    lines.extend(["## Arbre Textuel", "", "```text", tree_text, "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_index(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Decision Tree Rule Extraction",
        "",
        "Ces arbres sont des modeles interpretables peu profonds entraines pour extraire des regles.",
        "Ils completent les modeles retenus principaux, qui restent les meilleurs modeles predictifs.",
        "",
        "## Methode",
        "",
        "- Donnees : tables `modeling/under_dataset.csv` et `modeling/over_dataset.csv`.",
        "- Split : groupe par `schema_id`, environ 70% train, 15% validation, 15% test.",
        "- Features : liste retenue du modele principal correspondant.",
        "- Candidats : arbres de profondeur 2 a 5 et `min_samples_leaf` 20, 50, 100.",
        "- Selection : PR-AUC validation, puis F1, recall, precision.",
        "- Seuil : choisi sur validation pour maximiser le F1.",
        "",
        "## Resultats",
        "",
        "| framework | target | tree | features | F1 | precision | recall | PR-AUC | ROC-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['framework']} | {row['target'].upper()} | {row['model']} | {row['features']} | "
            f"{float(row['f1']):.3f} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | "
            f"{float(row['pr_auc']):.3f} | {float(row['roc_auc']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Fichiers",
            "",
            "Chaque dossier contient :",
            "",
            "- `*_tree.svg` : plot de l'arbre.",
            "- `*_tree.txt` : arbre textuel complet.",
            "- `*_positive_rules.csv` : feuilles positives sous forme de regles.",
            "- `*_all_leaf_rules.csv` : toutes les feuilles, positives et negatives.",
            "- `*_rules.md` : explication et regles lisibles.",
            "- `*_candidate_metrics.csv` : metriques de tous les arbres candidats.",
            "- `*_selected_metrics.csv` : metriques de l'arbre retenu.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def refresh_existing_markdown(coverage_root: Path, frameworks: list[str], targets: list[str]) -> None:
    output_root = coverage_root / "rules"
    summary_path = output_root / "decision_tree_rule_summary.csv"
    summaries = read_csv(summary_path) if summary_path.exists() else []
    if summaries:
        write_index(output_root / "README.md", summaries)

    for framework in frameworks:
        for target in targets:
            target_root = output_root / framework / target
            metrics_path = target_root / f"{target}_selected_metrics.csv"
            rules_path = target_root / f"{target}_all_leaf_rules.csv"
            tree_path = target_root / f"{target}_tree.txt"
            if not metrics_path.exists() or not rules_path.exists() or not tree_path.exists():
                continue
            metrics_rows = read_csv(metrics_path)
            rules = read_csv(rules_path)
            if not metrics_rows:
                continue
            metrics = metrics_rows[0]
            model_name = str(metrics.get("model", "selected_tree"))
            threshold = float(metrics.get("threshold", 0.5))
            tree_text = tree_path.read_text(encoding="utf-8")
            write_rules_markdown(
                target_root / f"{target}_rules.md",
                framework,
                target,
                model_name,
                threshold,
                metrics,
                rules,
                tree_text,
            )
            print(f"refreshed: {target_root / f'{target}_rules.md'}")


def run_target(
    framework_root: Path,
    framework: str,
    target: str,
    output_root: Path,
    max_depths: list[int],
    min_samples_leaf_values: list[int],
    seed: int,
    validation_size: float,
    test_size: float,
) -> dict[str, Any] | None:
    data_path = framework_root / "modeling" / f"{target}_dataset.csv"
    feature_path = framework_root / "modeling" / f"selected_{target}_features.txt"
    if not data_path.exists() or not feature_path.exists():
        return None
    rows = read_csv(data_path)
    label = f"y_{target}"
    if not rows or label not in rows[0] or len({int(row[label]) for row in rows}) < 2:
        return None
    features = read_feature_list(feature_path)
    splits = split_rows(rows, seed + (1 if target == "over" else 0), validation_size, test_size)
    train_rows = splits["train"]
    validation_rows = splits["validation"]
    test_rows = splits["test"]

    candidates: list[tuple[Pipeline, float, dict[str, Any]]] = []
    metric_rows: list[dict[str, Any]] = []
    for max_depth in max_depths:
        for min_samples_leaf in min_samples_leaf_values:
            pipeline, threshold, validation_metrics = fit_candidate(
                train_rows,
                validation_rows,
                features,
                label,
                max_depth,
                min_samples_leaf,
                seed,
            )
            x_test = make_feature_matrix(test_rows, features)
            y_test = np.asarray([int(row[label]) for row in test_rows], dtype=int)
            test_score = pipeline.predict_proba(x_test)[:, 1]
            model_name = str(validation_metrics["model"])
            test_metrics = sklearn_metrics_row(model_name, "test", y_test, test_score, threshold=threshold)
            test_confusion = sklearn_confusion_row(model_name, "test", y_test, test_score, threshold=threshold)
            metric_rows.append({"framework": framework, "target": target, **validation_metrics})
            metric_rows.append({"framework": framework, "target": target, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, **test_metrics})
            metric_rows.append({"framework": framework, "target": target, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, **test_confusion})
            candidates.append((pipeline, threshold, validation_metrics))

    best_pipeline, best_threshold, best_validation = max(candidates, key=lambda item: model_score(item[2]))
    model_name = str(best_validation["model"])
    x_test = make_feature_matrix(test_rows, features)
    y_test = np.asarray([int(row[label]) for row in test_rows], dtype=int)
    test_score = best_pipeline.predict_proba(x_test)[:, 1]
    best_test_metrics = sklearn_metrics_row(model_name, "test", y_test, test_score, threshold=best_threshold)
    best_test_confusion = sklearn_confusion_row(model_name, "test", y_test, test_score, threshold=best_threshold)

    target_root = output_root / framework / target
    target_root.mkdir(parents=True, exist_ok=True)
    write_csv(target_root / f"{target}_candidate_metrics.csv", metric_rows)
    write_csv(target_root / f"{target}_selected_metrics.csv", [{**best_test_metrics, **best_test_confusion}])

    names = transformed_feature_names(best_pipeline, features)
    tree_text = export_text(best_pipeline.named_steps["classifier"], feature_names=names, max_depth=12)
    (target_root / f"{target}_tree.txt").write_text(tree_text, encoding="utf-8")
    plot_tree_svg(
        target_root / f"{target}_tree.svg",
        best_pipeline,
        features,
        target,
        f"{framework} {target.upper()} - {model_name}",
        best_threshold,
    )
    rules = leaf_rules(best_pipeline, features, best_threshold, target)
    add_test_rule_metrics(rules, best_pipeline, features, test_rows, label)
    positive_rules = [rule for rule in rules if rule["predicted_class"] == target.upper()]
    write_csv(
        target_root / f"{target}_positive_rules.csv",
        positive_rules,
        [
            "target",
            "leaf_id",
            "predicted_class",
            "positive_rate_train_leaf",
            "positive_count_train_leaf_weighted",
            "negative_count_train_leaf_weighted",
            "weighted_support_train_leaf",
            "n_conditions",
            "test_support",
            "test_positives",
            "test_precision",
            "test_recall_contribution",
            "rule",
        ],
    )
    write_csv(
        target_root / f"{target}_all_leaf_rules.csv",
        rules,
        [
            "target",
            "leaf_id",
            "predicted_class",
            "positive_rate_train_leaf",
            "positive_count_train_leaf_weighted",
            "negative_count_train_leaf_weighted",
            "weighted_support_train_leaf",
            "n_conditions",
            "test_support",
            "test_positives",
            "test_precision",
            "test_recall_contribution",
            "rule",
        ],
    )
    write_rules_markdown(
        target_root / f"{target}_rules.md",
        framework,
        target,
        model_name,
        best_threshold,
        best_test_metrics,
        rules,
        tree_text,
    )
    return {
        "framework": framework,
        "target": target,
        "model": model_name,
        "features": len(features),
        **best_test_metrics,
        **best_test_confusion,
    }


def main() -> None:
    args = parse_args()
    coverage_root = Path(args.coverage_root)
    if args.refresh_markdown_only:
        refresh_existing_markdown(coverage_root, args.frameworks, args.targets)
        return
    output_root = coverage_root / "rules"
    summaries: list[dict[str, Any]] = []
    for framework in args.frameworks:
        framework_root = coverage_root / "modeles_predictifs" / framework
        for target in args.targets:
            summary = run_target(
                framework_root,
                framework,
                target,
                output_root,
                args.max_depths,
                args.min_samples_leaf,
                args.seed,
                args.validation_size,
                args.test_size,
            )
            if summary is not None:
                summaries.append(summary)
                print(
                    f"{framework} {target}: {summary['model']} "
                    f"F1={float(summary['f1']):.3f} P={float(summary['precision']):.3f} R={float(summary['recall']):.3f}"
                )
            else:
                print(f"{framework} {target}: skipped")
    write_csv(output_root / "decision_tree_rule_summary.csv", summaries)
    write_index(output_root / "README.md", summaries)
    print(f"done: {output_root}")


if __name__ == "__main__":
    main()
