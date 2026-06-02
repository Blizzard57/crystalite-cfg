#!/usr/bin/env python3
"""Measure classifier-free-guidance *conditioning adherence*.

The repo's `eval_crystalite_ckpt.py` already measures generation *quality*
(validity / novelty / uniqueness / S.U.N / Wasserstein / thermo). It does not
measure whether conditioned samples actually hit their target property. This
script fills that gap for the directly-verifiable properties:

  * space_group     -> realized via pymatgen SpacegroupAnalyzer
  * chemical_system -> realized via the structure's element set

Both are read straight from the generated structures, so no property predictor
or DFT is needed. (Scalar properties such as band_gap / dft_mag_density require
an external predictor or DFT; energy_above_hull can be scored with the repo's
existing thermo stack via `eval_crystalite_ckpt.py --thermo_*`.)

Typical workflow (mirrors MatterGen's "generate -> score adherence"):

  # 1. generate conditioned samples
  python src/sample_crystalite_ckpt.py --checkpoint <ckpt> \
      --target space_group=225 --guidance_scale 2.0 \
      --num_samples 256 --save_pt --output_dir outputs/cond/sg225

  # 2. score how well they hit the target
  python scripts/eval_conditioning.py \
      --samples outputs/cond/sg225/samples.pt --target space_group=225
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.mp20_tokens import tokens_to_structure


def _parse_targets(raw: list[str]) -> dict[str, object]:
    targets: dict[str, object] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(f"Invalid --target '{entry}', expected NAME=VALUE.")
        name, value = entry.split("=", 1)
        try:
            targets[name.strip()] = float(value)
        except ValueError:
            targets[name.strip()] = value.strip()
    return targets


def _build_structures(samples: list[dict]):
    """Reconstruct pymatgen Structures; return (structures, n_failed)."""
    structures, failed = [], 0
    for item in samples:
        try:
            structures.append(tokens_to_structure(item))
        except Exception:
            failed += 1
    return structures, failed


def _space_group_metrics(structures, target: int, symprec: float) -> dict:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    realized, errors = [], 0
    for s in structures:
        try:
            realized.append(SpacegroupAnalyzer(s, symprec=symprec).get_space_group_number())
        except Exception:
            errors += 1
    if not realized:
        return {"space_group/analyzable": 0.0}
    realized_t = torch.tensor(realized, dtype=torch.float64)
    target_t = float(int(target))
    return {
        "space_group/target": target_t,
        "space_group/analyzed": float(len(realized)),
        "space_group/analyzer_errors": float(errors),
        "space_group/match_rate": float((realized_t == target_t).float().mean()),
        "space_group/mae": float((realized_t - target_t).abs().mean()),
    }


def _chemical_system_metrics(structures, target: str) -> dict:
    target_set = set(target.split("-"))
    subset_hits, exact_hits = 0, 0
    for s in structures:
        els = {e.symbol for e in s.composition.elements}
        if els <= target_set:
            subset_hits += 1
        if els == target_set:
            exact_hits += 1
    n = max(len(structures), 1)
    return {
        "chemical_system/target": target,
        "chemical_system/subset_rate": subset_hits / n,  # no foreign elements
        "chemical_system/exact_rate": exact_hits / n,     # uses exactly the target set
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="samples.pt from sampling.")
    parser.add_argument(
        "--target", action="append", default=[], metavar="NAME=VALUE",
        help="The conditioning target(s) used to generate these samples. Repeatable.",
    )
    parser.add_argument(
        "--symprec", type=float, default=0.1,
        help="Symmetry tolerance for space-group analysis (pymatgen default 0.01).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    targets = _parse_targets(args.target)
    samples = torch.load(args.samples, weights_only=False)
    structures, n_failed = _build_structures(samples)

    metrics: dict[str, float | str] = {
        "num_samples": float(len(samples)),
        "buildable": float(len(structures)),
        "buildable_rate": len(structures) / max(len(samples), 1),
    }
    for name, value in targets.items():
        if name == "space_group":
            metrics.update(_space_group_metrics(structures, int(value), args.symprec))
        elif name == "chemical_system":
            metrics.update(_chemical_system_metrics(structures, str(value)))
        else:
            metrics[f"{name}/note"] = (
                "scalar property: not directly verifiable from structure; use an external "
                "predictor/DFT, or eval_crystalite_ckpt.py --thermo_* for energy_above_hull."
            )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
