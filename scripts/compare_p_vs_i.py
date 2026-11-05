"""Compare a P-chain run and an I-only run on matched PSNR.

Both inputs are JSON files produced by eval_forward_video.py.

Example:
    python scripts/compare_p_vs_i.py \
        --p_json outputs/p1/hevc_d_dense.json \
        --i_json outputs/p1/hevc_d_intra_dense.json \
        --plot outputs/p1/p_vs_i.png
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_points(path, force_intra):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = []
    for key, res in data["results"].items():
        s = res["summary"]
        pts.append({
            "key": key,
            "qp_i": res.get("qp_i"),
            "qp_p": res.get("qp_p"),
            "bpp": float(s["bpp_all"]),
            "psnr": float(s["psnr_all"]),
            "psnr_y": float(s.get("psnr_y_all", 0.0)),
        })
    pts.sort(key=lambda x: x["psnr"])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_json", required=True)
    ap.add_argument("--i_json", required=True)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    p = load_points(args.p_json, force_intra=False)
    i = load_points(args.i_json, force_intra=True)

    i_psnr = np.array([x["psnr"] for x in i])
    i_bpp = np.array([x["bpp"] for x in i])

    print("P-chain points")
    print("  qp_i qp_p    bpp     psnr")
    for x in p:
        print(f"  {x['qp_i']:4d} {x['qp_p']:4d}  {x['bpp']:7.4f}  {x['psnr']:7.3f}")

    print("\nI-only points")
    print("  qp_i        bpp     psnr")
    for x in i:
        print(f"  {x['qp_i']:4d}       {x['bpp']:7.4f}  {x['psnr']:7.3f}")

    print("\nMatched-PSNR comparison (linear interpolation on the I-only curve)")
    print("  qp_i qp_p   P bpp   P psnr   I bpp@psnr   saving")
    savings = []
    for x in p:
        if x["psnr"] < i_psnr[0] or x["psnr"] > i_psnr[-1]:
            print(f"  {x['qp_i']:4d} {x['qp_p']:4d}  {x['bpp']:7.4f}  "
                  f"{x['psnr']:7.3f}   (out of I range)")
            continue
        i_at = float(np.interp(x["psnr"], i_psnr, i_bpp))
        saving = 1.0 - x["bpp"] / i_at
        savings.append(saving)
        print(f"  {x['qp_i']:4d} {x['qp_p']:4d}  {x['bpp']:7.4f}  "
              f"{x['psnr']:7.3f}   {i_at:9.4f}   {saving * 100:6.1f}%")

    if savings:
        print(f"\nAverage matched-PSNR bpp saving: "
              f"{100.0 * float(np.mean(savings)):.1f}%  "
              f"(over {len(savings)} points)")
        print(f"Min / max saving: {100.0 * min(savings):.1f}% / "
              f"{100.0 * max(savings):.1f}%")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[warn] matplotlib unavailable: {e}")
            return

        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4.5))
        plt.plot([x["bpp"] for x in i], [x["psnr"] for x in i],
                 "o-", label="I-only")
        plt.plot([x["bpp"] for x in p], [x["psnr"] for x in p],
                 "s-", label="P-chain (HTS)")
        plt.xscale("log")
        plt.xlabel("bpp")
        plt.ylabel("PSNR (RGB-weighted)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
