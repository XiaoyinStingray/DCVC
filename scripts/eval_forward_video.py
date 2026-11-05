"""Forward-only evaluation of the DCVC-UF video model (I frame + P chunks).

No CUDA extension / bitstream: it calls `forward_one_frame` and computes
YUV420 PSNR + bpp. This is the P1 video anchor and the basis for the
P-vs-I / reset probes.

Example (HEVC-D, 96 frames, HT-S):

    python scripts/eval_forward_video.py \
        --src_type yuv420 \
        --src_path /gdata1/zhaojy/datasets/test/HEVC/D/BasketballPass_416x240_50.yuv \
        --width 416 --height 240 --frames 96 \
        --model_path_i checkpoints/cvpr2026_image.pth.tar \
        --model_path_p checkpoints/cvpr2026_video_hts.pth.tar \
        --model_structure hts \
        --qp_i 0 63 --qp_p 0 63 \
        --out outputs/p1/hevc_d.json

I-only baseline:

    ... --force_intra
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.image_model import DMCI  # noqa: E402
from src.models.video_model_ht import DMC as DMC_HT, g_frame_delay as DELAY_HT  # noqa: E402
from src.models.video_model_ld import DMC as DMC_LD, g_frame_delay as DELAY_LD  # noqa: E402
from src.utils.common import ModelStructure, get_state_dict  # noqa: E402
from src.utils.metrics import calc_psnr  # noqa: E402
from src.utils.transforms import (  # noqa: E402
    rgb2ycbcr, ycbcr2rgb, ycbcr420_to_444_np, yuv_444_to_420,
)
from src.utils.video_reader import PNGReader, YUV420Reader  # noqa: E402


def load_checkpoint(path):
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


def load_net(net, path, device, half, random_init=False):
    if not random_init:
        state = load_checkpoint(path)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        if not missing and not unexpected:
            print(f"[ok] loaded {len(state)} tensors from {path}")
    net.to(device).eval()
    if half:
        net.half()
    return net


def make_reader(args):
    if args.src_type == "yuv420":
        return YUV420Reader(args.src_path, args.width, args.height)
    return PNGReader(args.src_path, args.width, args.height)


def read_frame(reader, args):
    """Return (x [3,H,W] in [-0.5,0.5], original payload) or None at EOF."""
    if args.src_type == "yuv420":
        y, uv = reader.read_one_frame()
        if y is None:
            return None
        yuv444 = ycbcr420_to_444_np(y, uv)  # [3,H,W] uint8
        x = torch.from_numpy(yuv444).float() / 255.0 - 0.5
        return x, (y, uv)

    rgb = reader.read_one_frame()
    if rgb is None:
        return None
    x = torch.from_numpy(rgb).float() / 255.0
    x = rgb2ycbcr(x.unsqueeze(0))[0] - 0.5  # [3,H,W]
    return x, rgb


def pad_to_16(x, pr, pb, mode):
    if pr == 0 and pb == 0:
        return x
    return F.pad(x.unsqueeze(0), (0, pr, 0, pb), mode=mode).squeeze(0)


def metrics(args, x_hat_crop, orig):
    """x_hat_crop: [1,3,h,w] in [-0.5,0.5]; orig: (y,uv) or rgb [3,h,w]."""
    x_hat_crop = x_hat_crop.float()
    if args.src_type == "yuv420":
        y, uv = orig
        y_rec, uv_rec = yuv_444_to_420(x_hat_crop + 0.5)
        y_rec = torch.clamp(y_rec * 255.0, 0, 255).squeeze(0).cpu().numpy()
        uv_rec = torch.clamp(uv_rec * 255.0, 0, 255).squeeze(0).cpu().numpy()
        py = calc_psnr(y[0], y_rec[0])
        pu = calc_psnr(uv[0], uv_rec[0])
        pv = calc_psnr(uv[1], uv_rec[1])
        return (6 * py + pu + pv) / 8.0, py, pu, pv

    rec = ycbcr2rgb(x_hat_crop + 0.5).clamp(0.0, 1.0)
    rec_np = rec[0].permute(1, 2, 0).cpu().numpy() * 255.0
    org = orig.transpose(1, 2, 0).astype(np.float32)
    p = calc_psnr(org, rec_np)
    return p, p, 0.0, 0.0


def run_sequence(args, i_net, p_net, delay, qp_i, qp_p, device):
    reader = make_reader(args)
    h, w = args.height, args.width
    pr, pb = DMCI.get_padding_size(h, w, 64)
    mode = args.pad_mode

    frames = []
    for _ in range(args.frames):
        item = read_frame(reader, args)
        if item is None:
            break
        frames.append(item)
    reader.close()
    if not frames:
        raise SystemExit("no frames read")
    n = len(frames)
    print(f"  read {n} frames ({w}x{h}, pad {pr}x{pb})")

    qp_i_t = torch.tensor([qp_i], dtype=torch.long, device=device)
    qp_p_t = torch.tensor([qp_p], dtype=torch.long, device=device)

    def do_i_frame(idx):
        x, orig = frames[idx]
        x_pad = pad_to_16(x, pr, pb, mode).unsqueeze(0).to(device)
        if args.half:
            x_pad = x_pad.half()
        with torch.no_grad():
            res = i_net.forward_one_frame(x_pad, qp_i_t)
        x_hat_pad = res["x_hat"].float()
        x_hat = x_hat_pad[:, :, :h, :w]
        bits = float(res["bits_y"].item() + res["bits_z"].item())
        p, py, pu, pv = metrics(args, x_hat, orig)
        return {
            "idx": idx, "type": "I", "bpp": bits / (h * w),
            "bits": bits, "psnr": float(p), "psnr_y": float(py),
            "psnr_u": float(pu), "psnr_v": float(pv),
        }, x_hat_pad

    records = []
    bits_total = 0.0
    bits_i = 0.0
    bits_p = 0.0

    if args.force_intra:
        for idx in range(n):
            rec, _ = do_i_frame(idx)
            records.append(rec)
            bits_total += rec["bits"]
            bits_i += rec["bits"]
            if args.verbose:
                print(f"  {idx:3d} I bpp={rec['bpp']:.4f} psnr={rec['psnr']:.3f}")
        bpp_p = 0.0
        psnr_p = 0.0
    else:
        # first frame: I
        rec, x_hat_pad = do_i_frame(0)
        records.append(rec)
        bits_total += rec["bits"]
        bits_i += rec["bits"]
        if args.verbose:
            print(f"  {0:3d} I bpp={rec['bpp']:.4f} psnr={rec['psnr']:.3f}")

        # initial DPB from the I-frame reconstruction
        p_net.clear_dpb()
        ref = F.pixel_unshuffle(x_hat_pad, 8)
        if args.half:
            ref = ref.half()
        p_net.ref_feature = ref
        p_net.memory = None
        p_net.ctx = None

        frame_idx = 1
        while frame_idx < n:
            valid = min(delay, n - frame_idx)
            chunk_items = frames[frame_idx:frame_idx + valid]
            chunk_orig = [item[1] for item in chunk_items]
            chunk_frames = [item[0] for item in chunk_items]
            if valid < delay:
                chunk_frames += [chunk_frames[-1]] * (delay - valid)

            chunk = torch.cat([pad_to_16(x, pr, pb, mode) for x in chunk_frames],
                              dim=0).unsqueeze(0).to(device)
            if args.half:
                chunk = chunk.half()

            if args.reset_interval > 0:
                reset = 1 if (frame_idx + delay) % args.reset_interval == 1 else 0
            else:
                reset = 0

            with torch.no_grad():
                res = p_net.forward_one_frame(chunk, qp_p_t,
                                              reset_feature_memory=reset)
            x_hats = res["x_hat"] if isinstance(res["x_hat"], list) else [res["x_hat"]]
            bits = float(res["bits_y"].item() + res["bits_z"].item())
            bpp_per_frame = bits / (delay * h * w)
            bits_total += bits
            bits_p += bits

            for j in range(valid):
                x_hat = x_hats[j].float()[:, :, :h, :w]
                p, py, pu, pv = metrics(args, x_hat, chunk_orig[j])
                records.append({
                    "idx": frame_idx + j, "type": "P", "bpp": bpp_per_frame,
                    "psnr": float(p), "psnr_y": float(py),
                    "psnr_u": float(pu), "psnr_v": float(pv),
                })
                if args.verbose:
                    print(f"  {frame_idx + j:3d} P bpp={bpp_per_frame:.4f} "
                          f"psnr={p:.3f}")
            frame_idx += valid

        psnr_p = float(np.mean([r["psnr"] for r in records if r["type"] == "P"]))
        bpp_p = bits_p / (max(1, sum(1 for r in records if r["type"] == "P")) * h * w)

    psnr_i = float(np.mean([r["psnr"] for r in records if r["type"] == "I"]))
    psnr_all = float(np.mean([r["psnr"] for r in records]))
    psnr_y_all = float(np.mean([r["psnr_y"] for r in records]))
    bpp_all = bits_total / (len(records) * h * w)
    bpp_i = bits_i / (max(1, sum(1 for r in records if r["type"] == "I")) * h * w)

    summary = {
        "n_frames": len(records),
        "bpp_all": bpp_all,
        "bpp_i": bpp_i,
        "bpp_p": bpp_p,
        "psnr_all": psnr_all,
        "psnr_y_all": psnr_y_all,
        "psnr_i": psnr_i,
        "psnr_p": psnr_p,
    }
    return {"summary": summary, "frames": records}


def parse_args():
    p = argparse.ArgumentParser(description="Forward-only DCVC-UF video evaluation")
    p.add_argument("--src_type", type=str, default="yuv420",
                   choices=["yuv420", "png"])
    p.add_argument("--src_path", type=str, required=True,
                   help="YUV file (yuv420) or PNG frame directory (png)")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--model_path_i", type=str, required=True)
    p.add_argument("--model_path_p", type=str, default=None)
    p.add_argument("--model_structure", type=str, default="hts",
                   choices=["hts", "htl", "ld"])
    p.add_argument("--qp_i", type=int, nargs="+", default=[0, 21, 42, 63])
    p.add_argument("--qp_p", type=int, nargs="+", default=[0, 21, 42, 63])
    p.add_argument("--force_intra", action="store_true")
    p.add_argument("--reset_interval", type=int, default=128)
    p.add_argument("--out", type=str, default="outputs/p1/video.json")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--half", action="store_true")
    p.add_argument("--pad_mode", type=str, default="replicate",
                   choices=["replicate", "constant"])
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.force_intra and not args.model_path_p:
        raise SystemExit("--model_path_p is required unless --force_intra is set")
    if len(args.qp_i) != len(args.qp_p) and not args.force_intra:
        raise SystemExit("--qp_i and --qp_p must have the same length")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {device} | structure: {args.model_structure} | "
          f"frames: {args.frames} | force_intra: {args.force_intra}")

    i_net = load_net(DMCI(), args.model_path_i, device, args.half)

    if args.force_intra:
        p_net = None
        delay = 1
    elif args.model_structure == "ld":
        p_net = load_net(DMC_LD(), args.model_path_p, device, args.half)
        delay = DELAY_LD
    else:
        p_net = load_net(DMC_HT(model_structure=ModelStructure(args.model_structure)),
                         args.model_path_p, device, args.half)
        delay = DELAY_HT
    print(f"frame_delay: {delay}")

    results = {}
    for qi, qp in zip(args.qp_i, args.qp_p if not args.force_intra else args.qp_i):
        key = f"{qi}_{qp}" if not args.force_intra else f"{qi}_intra"
        print(f"\n[qp_i={qi} qp_p={qp}]")
        t0 = time.time()
        res = run_sequence(args, i_net, p_net, delay, qi, qp, device)
        res["qp_i"] = qi
        res["qp_p"] = None if args.force_intra else qp
        results[key] = res
        s = res["summary"]
        print(f"  bpp all/I/P: {s['bpp_all']:.4f} / {s['bpp_i']:.4f} / "
              f"{s['bpp_p']:.4f}")
        print(f"  psnr all/I/P: {s['psnr_all']:.3f} / {s['psnr_i']:.3f} / "
              f"{s['psnr_p']:.3f}  (Y all {s['psnr_y_all']:.3f})")
        print(f"  elapsed: {time.time() - t0:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "src_path": args.src_path,
        "src_type": args.src_type,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "model_structure": args.model_structure,
        "force_intra": bool(args.force_intra),
        "reset_interval": args.reset_interval,
        "model_path_i": args.model_path_i,
        "model_path_p": args.model_path_p,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
