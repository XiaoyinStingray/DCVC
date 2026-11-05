"""Forward-only evaluation of the DCVC-UF intra model (DMCI) on PNG images.

No CUDA extension, no bitstream, no proxy: it only calls
`DMCI.forward_one_frame` and computes RGB PSNR + bpp. This is the P1
"numerical anchor" before touching test_video.py.

Example (from the repo root):

    python scripts/eval_forward_image.py \
        --model_path_i checkpoints/cvpr2026_image.pth.tar \
        --images E:/code/dl/data/kodak \
        --qp 0 21 42 63 \
        --out outputs/p1/kodak.json

Smoke test:

    python scripts/eval_forward_image.py ... --limit 2 --qp 0 63
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.image_model import DMCI  # noqa: E402
from src.utils.common import get_state_dict  # noqa: E402
from src.utils.metrics import calc_psnr  # noqa: E402
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb  # noqa: E402


def load_checkpoint(path):
    """Official loader first; fall back to weights_only=False for old ckpts."""
    try:
        return get_state_dict(path)
    except Exception as e:
        print(f"[warn] get_state_dict failed ({e}); falling back to torch.load")
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if "net" in ckpt:
            ckpt = ckpt["net"]
        consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
        return ckpt


def load_model(path, device, random_init=False, half=False):
    model = DMCI()
    if not random_init:
        state = load_checkpoint(path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        if not missing and not unexpected:
            print(f"[ok] loaded {len(state)} tensors from {path}")
    else:
        print("[warn] --random-init: weights are random; metrics are meaningless")
    model.to(device).eval()
    if half:
        model.half()
    return model


def load_png_rgb(path):
    """Return (rgb_np HxWx3 in [0,1], x tensor [1,3,H,W] in YCbCr-0.5)."""
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    x = rgb2ycbcr(x) - 0.5
    return rgb, x


@torch.no_grad()
def evaluate_one(model, rgb, x, qp, device, half, pad_mode):
    h, w = x.shape[-2:]
    padding_r, padding_b = DMCI.get_padding_size(h, w, 64)
    x_in = x.to(device)
    if half:
        x_in = x_in.half()
    if padding_r > 0 or padding_b > 0:
        x_in = F.pad(x_in, (0, padding_r, 0, padding_b), mode=pad_mode)

    qp_t = torch.tensor([qp], dtype=torch.long, device=device)
    out = model.forward_one_frame(x_in, qp_t)

    x_hat = out["x_hat"][:, :, :h, :w].float()
    rec = ycbcr2rgb(x_hat + 0.5).clamp(0.0, 1.0)
    rec_np = rec[0].permute(1, 2, 0).cpu().numpy()

    psnr = calc_psnr(rgb * 255.0, rec_np * 255.0, data_range=255.0)

    # Y/U/V PSNR (useful when comparing with YUV-oriented tables)
    x_crop = x[:, :, :h, :w].float()
    xh_cpu = x_hat.cpu()
    psnr_comp = {}
    for name, idx in (("y", 0), ("u", 1), ("v", 2)):
        org = (x_crop[0, idx] + 0.5).clamp(0, 1).numpy() * 255.0
        rec_c = (xh_cpu[0, idx] + 0.5).clamp(0, 1).numpy() * 255.0
        psnr_comp[name] = float(calc_psnr(org, rec_c))

    bits = float(out["bits_y"].item() + out["bits_z"].item())
    bpp = bits / (h * w)
    return {
        "bpp": bpp,
        "psnr": float(psnr),
        "psnr_y": psnr_comp["y"],
        "psnr_u": psnr_comp["u"],
        "psnr_v": psnr_comp["v"],
        "bits_y": float(out["bits_y"].item()),
        "bits_z": float(out["bits_z"].item()),
        "height": h,
        "width": w,
    }, rec_np


def parse_args():
    p = argparse.ArgumentParser(description="Forward-only DMCI evaluation on PNGs")
    p.add_argument("--model_path_i", type=str, default=None,
                   help="image checkpoint; ignored with --random-init")
    p.add_argument("--images", type=str, required=True,
                   help="folder containing PNG images")
    p.add_argument("--pattern", type=str, default="*.png")
    p.add_argument("--qp", type=int, nargs="+", default=[0, 21, 42, 63])
    p.add_argument("--out", type=str, default="outputs/p1/kodak.json")
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all images; N = first N sorted images")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--half", action="store_true")
    p.add_argument("--pad_mode", type=str, default="replicate",
                   choices=["replicate", "constant"])
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--save-recon", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.random_init and not args.model_path_i:
        raise SystemExit("--model_path_i is required unless --random-init is set")

    image_dir = Path(args.images)
    if not image_dir.is_dir():
        raise SystemExit(f"not a directory: {image_dir}")
    paths = sorted(image_dir.glob(args.pattern))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images matching {args.pattern} in {image_dir}")

    print(f"device: {device} | images: {len(paths)} | qp: {args.qp}")
    model = load_model(args.model_path_i, device,
                       random_init=args.random_init, half=args.half)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recon_dir = out_path.parent / "recon"
    if args.save_recon:
        recon_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    per_qp = {qp: {"bpp": [], "psnr": [], "psnr_y": []} for qp in args.qp}

    for idx, path in enumerate(paths):
        rgb, x = load_png_rgb(path)
        name = path.name
        results[name] = {}
        for qp in args.qp:
            torch.manual_seed(args.seed + idx * 1000 + qp)
            res, rec_np = evaluate_one(
                model, rgb, x, qp, device, args.half, args.pad_mode)
            results[name][str(qp)] = res
            per_qp[qp]["bpp"].append(res["bpp"])
            per_qp[qp]["psnr"].append(res["psnr"])
            per_qp[qp]["psnr_y"].append(res["psnr_y"])
            if args.save_recon:
                rec_img = Image.fromarray(
                    np.clip(rec_np * 255.0 + 0.5, 0, 255).astype(np.uint8))
                rec_img.save(recon_dir / f"{path.stem}_q{qp}.png")
            if args.verbose:
                print(f"  {name} qp={qp:2d} bpp={res['bpp']:.4f} "
                      f"psnr={res['psnr']:.3f}")

    summary = {}
    print("\n  qp      bpp   psnr_rgb    psnr_y    n")
    print("  " + "-" * 40)
    for qp in args.qp:
        bpp = float(np.mean(per_qp[qp]["bpp"]))
        psnr = float(np.mean(per_qp[qp]["psnr"]))
        psnr_y = float(np.mean(per_qp[qp]["psnr_y"]))
        summary[str(qp)] = {"bpp": bpp, "psnr": psnr, "psnr_y": psnr_y,
                            "n": len(per_qp[qp]["bpp"])}
        print(f"  {qp:2d}  {bpp:8.4f}  {psnr:7.3f}  {psnr_y:7.3f}  "
              f"{len(per_qp[qp]['bpp']):3d}")

    payload = {
        "model_path": str(args.model_path_i),
        "random_init": bool(args.random_init),
        "half": bool(args.half),
        "images": results,
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
