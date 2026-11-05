"""Summarize the reference-counterfactual probe across several qp_p values.

Input: outputs/probe/ref_qp{0,21,42,63}.json from probe_ref.py.

Prints:
  1. content penalty (true vs far/black/gray) per qp_p
  2. quality curve (qp_i -> P psnr/bpp) per qp_p

Optional plot: outputs/probe/ref_summary.png
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_many(paths):
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        out.append(d)
    out.sort(key=lambda d: d["qp_p"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/probe/ref_qp*.json")
    ap.add_argument("--plot", default="outputs/probe/ref_summary.png")
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matching {args.glob}")
    runs = load_many(paths)

    print("=== content penalty (relative to true reference) ===")
    print("  qp_p  ref      dPSNR     bpp_ratio")
    content_rows = []
    for d in runs:
        qp = d["qp_p"]
        c = d["content"]
        base = c["true"]
        for name in ("far", "black", "gray"):
            r = c[name]
            d_psnr = r["psnr"] - base["psnr"]
            ratio = r["bpp"] / base["bpp"]
            content_rows.append((qp, name, d_psnr, ratio))
            print(f"  {qp:4d}  {name:5s}  {d_psnr:7.3f}   {ratio:7.3f}")

    print("\n=== quality curve (P chunk at fixed qp_p) ===")
    print("  qp_p  qp_i  ref_psnr   p_bpp    p_psnr   gain_vs_qp_i0")
    quality_rows = []
    for d in runs:
        qp = d["qp_p"]
        q = d["quality"]
        base = q["0"]["psnr"]
        for qi in sorted(q, key=lambda x: int(x)):
            r = q[qi]
            gain = r["psnr"] - base
            quality_rows.append((qp, int(qi), r["ref_psnr"], r["bpp"], r["psnr"], gain))
            print(f"  {qp:4d}  {qi:>4s}  {r['ref_psnr']:8.3f}  {r['bpp']:7.4f}  "
                  f"{r['psnr']:7.3f}   {gain:+7.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[warn] matplotlib unavailable: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for name, marker in (("far", "o"), ("black", "s"), ("gray", "^")):
        rows = [r for r in content_rows if r[1] == name]
        ax.plot([r[0] for r in rows], [r[2] for r in rows], marker + "-", label=name)
    ax.set_xlabel("qp_p")
    ax.set_ylabel("ΔPSNR vs true reference (dB)")
    ax.set_title("Wrong reference: quality penalty")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for d in runs:
        qp = d["qp_p"]
        q = d["quality"]
        xs = [q[k]["ref_psnr"] for k in sorted(q, key=lambda x: int(x))]
        ys = [q[k]["psnr"] for k in sorted(q, key=lambda x: int(x))]
        ax.plot(xs, ys, "o-", label=f"qp_p={qp}")
    ax.set_xlabel("reference PSNR (I-frame)")
    ax.set_ylabel("P-chunk PSNR")
    ax.set_title("Reference quality vs P quality")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = Path(args.plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
