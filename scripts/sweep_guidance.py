#!/usr/bin/env python3
"""Classifier-free-guidance sweep: generate at several guidance scales and score
conditioning adherence for each, then print/save one comparison table.

This is a thin orchestrator around the existing entrypoints
`src/sample_crystalite_ckpt.py` (generation) and `scripts/eval_conditioning.py`
(adherence) so there is no duplicated logic. The `guidance_scale=0` row is the
unconditional baseline; conditioning works only if `match_rate` climbs clearly
above it as the scale increases.

Example (single space-group target):
  python scripts/sweep_guidance.py \
      --checkpoint outputs/cond_mp20/checkpoints/final.pt \
      --dataset_name mp20 --data_root data/mp20 \
      --target space_group=225 --guidance_scales 0 2 4 \
      --num_samples 256 --output_dir outputs/sweep_sg225
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_name", default="mp20")
    p.add_argument("--data_root", default="data/mp20")
    p.add_argument("--target", action="append", default=[], metavar="NAME=VALUE",
                   help="Conditioning target(s); repeatable. Passed to both generation and scoring.")
    p.add_argument("--guidance_scales", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--sample_num_steps", type=int, default=150)
    p.add_argument("--sample_mode", default="regular", choices=["regular", "ema"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--symprec", type=float, default=0.1)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    if not args.target:
        p.error("at least one --target is required (e.g. --target space_group=225)")

    out_root = Path(args.output_dir)
    repo_root = Path(__file__).resolve().parents[1]
    target_flags: list[str] = []
    for t in args.target:
        target_flags += ["--target", t]

    rows: list[dict] = []
    for w in args.guidance_scales:
        run_dir = out_root / f"w{w:g}"
        sample_cmd = [
            sys.executable, str(repo_root / "src/sample_crystalite_ckpt.py"),
            "--checkpoint", args.checkpoint,
            "--dataset_name", args.dataset_name, "--data_root", args.data_root,
            "--sample_mode", args.sample_mode, "--device", args.device,
            "--num_samples", str(args.num_samples),
            "--sample_num_steps", str(args.sample_num_steps),
            "--atom_count_strategy", "empirical",
            "--guidance_scale", str(w), *target_flags,
            "--save_pt", "--output_dir", str(run_dir),
        ]
        if args.bf16:
            sample_cmd.append("--bf16")
        _run(sample_cmd)

        adherence_path = run_dir / "adherence.json"
        eval_cmd = [
            sys.executable, str(repo_root / "scripts/eval_conditioning.py"),
            "--samples", str(run_dir / "samples.pt"),
            "--symprec", str(args.symprec), *target_flags,
            "--out", str(adherence_path),
        ]
        _run(eval_cmd)

        metrics = json.loads(adherence_path.read_text())
        rows.append({"guidance_scale": w, **metrics})

    # Aggregate.
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sweep_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))

    # Pretty table over the most informative columns.
    cols = ["guidance_scale", "space_group/match_rate", "space_group/mae",
            "chemical_system/exact_rate", "chemical_system/subset_rate", "buildable_rate"]
    present = [c for c in cols if any(c in r for r in rows)]
    print("\n===== GUIDANCE SWEEP =====")
    print("  ".join(f"{c.split('/')[-1]:>14}" for c in present))
    for r in rows:
        print("  ".join(f"{r.get(c, ''):>14.4f}" if isinstance(r.get(c), (int, float))
                        else f"{'':>14}" for c in present))
    print(f"\n[save] {out_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
