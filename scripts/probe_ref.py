"""Reference counterfactual probe for DCVC-UF.

Two sub-probes on the same first P chunk (8 frames):

1. content: true I-frame reconstruction vs far frame vs black vs gray.
2. quality: use the I-frame reconstructed at qp_i = 0/21/42/63 as reference.

No CUDA extension; forward only.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eval_forward_video import (  # noqa: E402
    load_net, make_reader, read_frame, pad_to_16, metrics,
)
from src.models.image_model import DMCI  # noqa: E402
from src.models.video_model_ht import DMC as DMC_HT  # noqa: E402
from src.models.video_model_ld import DMC as DMC_LD  # noqa: E402
from src.utils.common import ModelStructure  # noqa: E402


@torch.no_grad()
def encode_i(i_net, x, pr, pb, mode, qp, device, half):
    x_pad = pad_to_16(x, pr, pb, mode).unsqueeze(0).to(device)
    if half:
        x_pad = x_pad.half()
    res = i_net.forward_one_frame(x_pad, torch.tensor([qp], dtype=torch.long, device=device))
    return res["x_hat"].float()


@torch.no_grad()
def encode_chunk(p_net, chunk_orig, pr, pb, mode, qp, device, half):
    chunk = torch.cat([pad_to_16(x, pr, pb, mode) for x in chunk_orig], dim=0)
    chunk = chunk.unsqueeze(0).to(device)
    if half:
        chunk = chunk.half()
    res = p_net.forward_one_frame(chunk, torch.tensor([qp], dtype=torch.long, device=device))
    return res


@torch.no_grad()
def run_ref(p_net, ref_pad, chunk_x, chunk_orig, args, pr, pb, mode, qp_p, device):
    p_net.clear_dpb()
    ref = F.pixel_unshuffle(ref_pad.float().to(device), 8)
    if args.half:
        ref = ref.half()
    p_net.ref_feature = ref
    p_net.memory = None
    p_net.ctx = None

    res = encode_chunk(p_net, chunk_x, pr, pb, mode, qp_p, device, args.half)
    x_hats = res["x_hat"] if isinstance(res["x_hat"], list) else [res["x_hat"]]
    h, w = args.height, args.width
    bits = float(res["bits_y"].item() + res["bits_z"].item())
    psnrs = []
    for j, xh in enumerate(x_hats):
        p, _, _, _ = metrics(args, xh.float()[:, :, :h, :w], chunk_orig[j])
        psnrs.append(float(p))
    return {
        "bpp": bits / (len(chunk_orig) * h * w),
        "psnr": sum(psnrs) / len(psnrs),
        "bits": bits,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Reference counterfactual probe")
    p.add_argument("--src_type", type=str, default="yuv420", choices=["yuv420", "png"])
    p.add_argument("--src_path", type=str, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--far_idx", type=int, default=95)
    p.add_argument("--model_path_i", type=str, required=True)
    p.add_argument("--model_path_p", type=str, required=True)
    p.add_argument("--model_structure", type=str, default="hts", choices=["hts", "htl", "ld"])
    p.add_argument("--qp_i", type=int, default=42)
    p.add_argument("--qp_p", type=int, default=42)
    p.add_argument("--qp_i_list", type=int, nargs="+", default=[0, 21, 42, 63])
    p.add_argument("--out", type=str, default="outputs/probe/ref.json")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--half", action="store_true")
    p.add_argument("--pad_mode", type=str, default="replicate", choices=["replicate", "constant"])
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    reader = make_reader(args)
    need = max(9, args.far_idx + 1)
    frames = []
    for _ in range(need):
        item = read_frame(reader, args)
        if item is None:
            break
        frames.append(item)
    reader.close()
    if len(frames) < 9:
        raise SystemExit(f"need at least 9 frames, got {len(frames)}")
    far_idx = min(args.far_idx, len(frames) - 1)
    print(f"device: {device} | frames: {len(frames)} | far_idx: {far_idx}")

    i_net = load_net(DMCI(), args.model_path_i, device, args.half)
    if args.model_structure == "ld":
        p_net = load_net(DMC_LD(), args.model_path_p, device, args.half)
    else:
        p_net = load_net(DMC_HT(model_structure=ModelStructure(args.model_structure)),
                         args.model_path_p, device, args.half)

    h, w = args.height, args.width
    pr, pb = DMCI.get_padding_size(h, w, 64)
    mode = args.pad_mode
    chunk_orig = [frames[i][1] for i in range(1, 9)]
    chunk_x = [frames[i][0] for i in range(1, 9)]

    # reference 0: true I-frame
    ref_true = encode_i(i_net, frames[0][0], pr, pb, mode, args.qp_i, device, args.half)
    # reference 1: far frame
    ref_far = encode_i(i_net, frames[far_idx][0], pr, pb, mode, args.qp_i, device, args.half)
    # reference 2/3: black / gray in YCbCr [-0.5, 0.5]
    hp, wp = ref_true.shape[-2:]
    black = torch.zeros(1, 3, hp, wp)
    black[:, 0] = -0.5
    gray = torch.zeros(1, 3, hp, wp)

    refs = {
        "true": ref_true,
        "far": ref_far,
        "black": black,
        "gray": gray,
    }

    content = {}
    print("\n[content]")
    print("  ref      ref_psnr    bpp      psnr")
    print("  " + "-" * 38)
    for name, ref in refs.items():
        ref_crop = ref.float()[:, :, :h, :w]
        ref_psnr, _, _, _ = metrics(args, ref_crop, frames[0][1])
        r = run_ref(p_net, ref, chunk_x, chunk_orig, args, pr, pb, mode, args.qp_p, device)
        r["ref_psnr"] = float(ref_psnr)
        content[name] = r
        print(f"  {name:6s}  {ref_psnr:8.3f}  {r['bpp']:8.4f}  {r['psnr']:8.3f}")

    quality = {}
    print("\n[quality]  (qp_p fixed)")
    print("  qp_i    ref_psnr    bpp      psnr")
    print("  " + "-" * 38)
    for qp_i in args.qp_i_list:
        ref = encode_i(i_net, frames[0][0], pr, pb, mode, qp_i, device, args.half)
        ref_psnr, _, _, _ = metrics(args, ref.float()[:, :, :h, :w], frames[0][1])
        r = run_ref(p_net, ref, chunk_x, chunk_orig, args, pr, pb, mode, args.qp_p, device)
        r["ref_psnr"] = float(ref_psnr)
        quality[str(qp_i)] = r
        print(f"  {qp_i:4d}  {ref_psnr:8.3f}  {r['bpp']:8.4f}  {r['psnr']:8.3f}")

    out = {
        "src_path": args.src_path,
        "model_structure": args.model_structure,
        "qp_i": args.qp_i,
        "qp_p": args.qp_p,
        "far_idx": far_idx,
        "content": content,
        "quality": quality,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
