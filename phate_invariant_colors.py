#!/usr/bin/env python3
"""Color solved-checkpoint PHATE coordinates by classical knot invariants.

Joins ``results/strategy_phate_prime3to8/knot_operator_phate.csv``
(job 24228941, width-192 checkpoint, 19-state pilot panel) with a KnotInfo
snapshot CSV (``name|...|determinant|...|signature|...`` pipe-delimited, as
shipped by database_knotinfo) and writes colored PNGs plus a joined CSV to
``results/strategy_phate_prime3to8/invariant_colored/``.

Usage:
    python phate_invariant_colors.py --knotinfo-csv path/to/knotinfo.csv

Exploratory figures only; no quantitative claim. Scalar invariants used here
are mirror-insensitive except signature, for which |signature| is plotted
(KnotInfo table hands may differ from corpus hands).
"""

import argparse
import csv
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PHATE_CSV = os.path.join(
    REPO_ROOT, "results", "strategy_phate_prime3to8",
    "knot_operator_phate.csv",
)
OUT_DIR = os.path.join(
    REPO_ROOT, "results", "strategy_phate_prime3to8", "invariant_colored"
)


def load_invariants(path):
    invariants = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name or "_" not in name:
                continue
            try:
                invariants[name] = {
                    "determinant": int(row["determinant"]),
                    "signature": int(row["signature"]),
                    "genus": int(row["three_genus"]),
                    "alternating": (row.get("alternating") or "").strip(),
                    "fibered": (row.get("fibered") or "").strip(),
                }
            except (ValueError, KeyError):
                continue
    return invariants


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knotinfo-csv", required=True)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    invariants = load_invariants(args.knotinfo_csv)
    points = []
    with open(PHATE_CSV, newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["knot"]
            if name not in invariants:
                print(f"WARNING: no invariants for {name}")
                continue
            points.append({
                "knot": name,
                "crossings": int(row["crossings"]),
                "x": float(row["phate_1"]),
                "y": float(row["phate_2"]),
                **invariants[name],
            })
    print(f"joined {len(points)}/35 knots")
    assert len(points) == 35, f"expected 35, got {len(points)}"

    with open(os.path.join(OUT_DIR, "phate_invariants.csv"), "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    for row in points:
        row["signature_abs"] = abs(row["signature"])
    numeric = [
        ("determinant", "Determinant", "viridis"),
        ("signature_abs", "|Signature| (mirror-insensitive)", "coolwarm"),
        ("genus", "Seifert genus", "plasma"),
    ]
    categorical = [
        ("alternating", "Alternating (Y/N)"),
        ("fibered", "Fibered (Y/N)"),
    ]
    for key, label in categorical:
        print(f"{key}: {sorted({row[key] for row in points})}")

    for key, label, cmap in numeric:
        fig, ax = plt.subplots(figsize=(7, 6))
        values = [row[key] for row in points]
        scatter = ax.scatter([row["x"] for row in points],
                             [row["y"] for row in points],
                             c=values, cmap=cmap, s=60,
                             edgecolors="black", linewidths=0.4)
        for row in points:
            ax.annotate(row["knot"], (row["x"], row["y"]), fontsize=6,
                        alpha=0.8)
        ax.set_title(f"Strategy-diffusion PHATE colored by {label}\n"
                     "exploratory: 19-state pilot, seed 0, no controls")
        ax.set_xlabel("PHATE 1")
        ax.set_ylabel("PHATE 2")
        fig.colorbar(scatter, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"phate_by_{key}.png"), dpi=150)
        plt.close(fig)

    for key, label in categorical:
        cats = sorted({row[key] for row in points})
        markers = {"Y": "o", "N": "s", "": "x"}
        fig, ax = plt.subplots(figsize=(7, 6))
        for cat in cats:
            xs = [row["x"] for row in points if row[key] == cat]
            ys = [row["y"] for row in points if row[key] == cat]
            ax.scatter(xs, ys, label=f"{cat or '?'} (n={len(xs)})",
                       s=70, marker=markers.get(cat, "d"),
                       edgecolors="black", linewidths=0.4)
        for row in points:
            ax.annotate(row["knot"], (row["x"], row["y"]), fontsize=6,
                        alpha=0.8)
        ax.set_title(f"Strategy-diffusion PHATE: {label}\n"
                     "exploratory: 19-state pilot, seed 0, no controls")
        ax.set_xlabel("PHATE 1")
        ax.set_ylabel("PHATE 2")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"phate_by_{key}.png"), dpi=150)
        plt.close(fig)
    print("wrote", sorted(os.listdir(OUT_DIR)))


if __name__ == "__main__":
    main()
