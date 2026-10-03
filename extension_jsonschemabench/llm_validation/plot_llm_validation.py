#!/usr/bin/env python3
"""Plot the baseline error rate of flagged vs control cases from results.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_llm_validation import OUT_DIR, ROOT, is_error  # noqa: E402

FIGURE = ROOT / "docs" / "figures" / "llm_validation_flagged_vs_control.svg"
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = [("flagged", "Flagged by a rule", "#2a78d6"), ("control", "Not flagged (control)", "#eb6834")]
TARGETS = [("over", "OVER\nvalid instance, Kubernetes"), ("under", "UNDER\ninvalid instance, GitHub")]


def main() -> None:
    records = [json.loads(line) for line in open(OUT_DIR / "results.jsonl", encoding="utf-8")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.3
    for offset, (group, label, color) in zip((-0.16, 0.16), GROUPS):
        for position, (target, _) in enumerate(TARGETS):
            items = [record for record in records if record["target"] == target and record["group"] == group]
            errors = sum(is_error(target, record["baseline"]) for record in items)
            rate = 100 * errors / len(items)
            ax.bar(position + offset, rate, width, color=color, label=label if position == 0 else None, zorder=2)
            ax.text(position + offset, rate + 2, f"{errors}/{len(items)}", ha="center", va="bottom", color=TEXT, fontsize=11)
    ax.set_xticks(range(len(TARGETS)), [label for _, label in TARGETS], color=TEXT, fontsize=10)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylabel("Cases where generation goes wrong", color=MUTED, fontsize=9)
    ax.set_title(
        "Rule-flagged cases fail far more often in real generation",
        loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=26,
    )
    ax.text(0, 1.035, "XGrammar + Qwen2.5-0.5B-Instruct, copy task, 15 cases per bar", transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(FIGURE, facecolor=SURFACE)
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
