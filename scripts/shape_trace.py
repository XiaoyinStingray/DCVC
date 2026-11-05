"""Shape / data-flow trace for DCVC-UF.

No checkpoint, no dataset, no CUDA extension needed.

It builds DMCI (I-frame codec) and DMC-HTS (8-frame chunk codec), runs one
forward pass on random tensors, prints the shape at each key module, checks the
expected shapes, and (optionally) runs backward to verify gradients.

Typical usage (from the repo root):

    python scripts/shape_trace.py --size 256
    python scripts/shape_trace.py --size 128 --no-backward       # small / CPU
    python scripts/shape_trace.py --model intra
    python scripts/shape_trace.py --model video --device cuda

Output is printed and also written to outputs/shape_trace/shape_trace.txt.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.image_model import DMCI  # noqa: E402
from src.models.video_model_ht import DMC, g_frame_delay  # noqa: E402
from src.utils.common import ModelStructure, loss_func  # noqa: E402


class Logger:
    def __init__(self, path):
        self.lines = []
        self.path = Path(path)

    def __call__(self, msg=""):
        print(msg)
        self.lines.append(str(msg))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\n[saved] {self.path}")


def fmt(x):
    if torch.is_tensor(x):
        return f"Tensor{tuple(x.shape)}"
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return "[]"
        if len(x) <= 3:
            return "[" + ", ".join(fmt(i) for i in x) + "]"
        return f"[{fmt(x[0])}, ... x{len(x)}]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {fmt(v)}" for k, v in x.items()) + "}"
    return type(x).__name__


def install_hooks(model, names, log, captured):
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            captured[name] = output
            log(f"  [{name}]")
            log(f"    in : {fmt(inputs)}")
            log(f"    out: {fmt(output)}")
        return hook

    for name in names:
        module = model.get_submodule(name)
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def check(name, actual, expected, log):
    if torch.is_tensor(actual):
        ok = tuple(actual.shape) == tuple(expected)
        got = tuple(actual.shape)
    elif isinstance(actual, (list, tuple)):
        ok = len(actual) == expected[0] and all(
            tuple(a.shape) == tuple(expected[1]) for a in actual)
        got = f"list[{len(actual)}] of {tuple(actual[0].shape)}" if actual else "empty"
    else:
        ok, got = False, type(actual).__name__
    log(f"  {'OK  ' if ok else 'FAIL'} {name}: got {got}, expected {expected}")
    return ok


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def trace_intra(args, log, device):
    log("\n" + "=" * 72)
    log("INTRA MODEL (DMCI): 1 frame")
    log("=" * 72)
    model = DMCI().to(device).train()
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    captured = {}
    names = ["enc", "hyper_enc", "hyper_dec", "y_prior_fusion", "dec"]
    handles = install_hooks(model, names, log, captured)

    x = torch.rand(1, 3, args.size, args.size, device=device) - 0.5
    qp = torch.tensor([args.qp], dtype=torch.long, device=device)
    lambdas = torch.tensor([args.lamda], dtype=torch.float32, device=device)

    log(f"\ninput: {fmt(x)}  qp={args.qp}  lambda={args.lamda}")
    t0 = time.time()
    res = model.forward_one_frame(x, qp)
    loss = loss_func(res, lambdas)
    log(f"forward time: {time.time() - t0:.2f}s")
    log(f"x_hat: {fmt(res['x_hat'])}")
    log(f"bpp: {res['bpp'].detach().cpu().numpy()}")
    log(f"mse: {res['mse'].detach().cpu().numpy()}")
    log(f"loss: {loss['loss'].detach().item():.4f}")

    h8, h16, h64 = args.size // 8, args.size // 16, args.size // 64
    check("y (enc out)", captured.get("enc"), (1, 256, h16, h16), log)
    check("z (hyper_enc out)", captured.get("hyper_enc"), (1, 128, h64, h64), log)
    check("x_hat", res["x_hat"], (1, 3, args.size, args.size), log)

    if args.backward:
        model.zero_grad(set_to_none=True)
        t0 = time.time()
        loss["loss"].backward()
        log(f"backward time: {time.time() - t0:.2f}s")
        log(f"grad norm: {grad_norm(model):.4f}")

    for h in handles:
        h.remove()
    return model


def trace_video(args, log, device, i_net=None):
    log("\n" + "=" * 72)
    log("VIDEO MODEL (DMC HTS): 8-frame chunk")
    log("=" * 72)
    model = DMC(model_structure=ModelStructure.HTS).to(device).train()
    log(f"frame_delay: {g_frame_delay}")
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    captured = {}
    names = ["feature_adaptor_i", "feature_adaptor_m", "feature_extractor",
             "encoder", "hyper_encoder", "hyper_decoder",
             "temporal_prior_encoder", "y_prior_fusion", "decoder", "recon_head"]
    handles = install_hooks(model, names, log, captured)

    ref = torch.rand(1, 3, args.size, args.size, device=device) - 0.5
    if i_net is not None:
        log("\n[reference] using the DMCI reconstruction of a random frame")
        with torch.inference_mode():
            ref = i_net.forward_one_frame(
                ref, torch.tensor([args.qp], dtype=torch.long, device=device),
                recon_only=True)
        log(f"  ref x_hat: {fmt(ref)}")

    chunk = torch.rand(1, 3 * g_frame_delay, args.size, args.size, device=device) - 0.5
    qp = torch.tensor([args.qp], dtype=torch.long, device=device)
    lambdas = torch.tensor([args.lamda], dtype=torch.float32, device=device)

    # initial state: I-frame -> pixel_unshuffle -> 192 channels @ H/8
    model.clear_dpb()
    model.ref_feature = F.pixel_unshuffle(ref, 8)
    log(f"\ninitial ref_feature: {fmt(model.ref_feature)}  (192 = 3*8*8)")
    log(f"initial memory/ctx : {model.memory} / {model.ctx}")

    t0 = time.time()
    res = model.forward_one_frame(chunk, qp)
    log(f"\npass 1 (normal) forward time: {time.time() - t0:.2f}s")
    log(f"  ref_feature after : {fmt(model.ref_feature)}  (decoder feature)")
    log(f"  memory used       : {fmt(model.memory)}")
    log(f"  ctx used          : {fmt(model.ctx)}")
    log(f"  x_hat             : {fmt(res['x_hat'])}")
    rd = model.get_rd_info(res)
    loss = loss_func(rd, lambdas)
    log(f"  bpp: {rd['bpp'].detach().cpu().numpy()}")
    log(f"  mse: {rd['mse'].detach().cpu().numpy()}")
    log(f"  loss: {loss['loss'].detach().item():.4f}")

    h8, h16, h64 = args.size // 8, args.size // 16, args.size // 64
    check("y (encoder out)", captured.get("encoder"), (1, 256, h16, h16), log)
    check("z (hyper_encoder out)", captured.get("hyper_encoder"), (1, 128, h64, h64), log)
    check("memory (used)", model.memory, (1, 512, h8, h8), log)
    check("ctx (used)", model.ctx, (1, 512, h8, h8), log)
    check("x_hat (8 frames)", res["x_hat"],
          (g_frame_delay, (1, 3, args.size, args.size)), log)

    if args.backward:
        model.zero_grad(set_to_none=True)
        t0 = time.time()
        loss["loss"].backward()
        log(f"\nbackward time: {time.time() - t0:.2f}s")
        log(f"grad norm: {grad_norm(model):.4f}")

    # detach the state before the next pass
    model.ref_feature = model.ref_feature.detach()
    if model.memory is not None:
        model.memory = model.memory.detach()

    # pass 2: normal start (FeatureAdaptorM), reset at the end
    with torch.no_grad():
        t0 = time.time()
        model.forward_one_frame(chunk, qp, reset_feature_memory=True)
        log(f"\npass 2 (reset at end) forward time: {time.time() - t0:.2f}s")
        log(f"  ref_feature after reset: {fmt(model.ref_feature)}  (192 = reset)")
        log(f"  memory after reset     : {model.memory}")
        log(f"  ctx after reset        : {model.ctx}")
        check("ref_feature after reset", model.ref_feature, (1, 192, h8, h8), log)

        # pass 3: after reset, the next adaptor should be FeatureAdaptorI
        model.apply_feature_adaptor()
        log("\npass 3 (after reset, adaptor only):")
        log(f"  memory after FeatureAdaptorI: {fmt(model.memory)}")
        log(f"  ctx after FeatureExtractor  : {fmt(model.ctx)}")
        check("memory after reset+adaptor", model.memory, (1, 512, h8, h8), log)

    for h in handles:
        h.remove()


def parse_args():
    p = argparse.ArgumentParser(description="DCVC-UF shape / data-flow trace")
    p.add_argument("--size", type=int, default=256,
                   help="spatial size; must be divisible by 64")
    p.add_argument("--qp", type=int, default=32)
    p.add_argument("--lamda", "--lambda", dest="lamda", type=float, default=100.0,
                   help="lambda for the RD loss used in backward")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--model", type=str, default="both",
                   choices=["both", "intra", "video"])
    p.add_argument("--backward", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--out", type=str,
                   default="outputs/shape_trace/shape_trace.txt")
    return p.parse_args()


def main():
    args = parse_args()
    if args.size % 64 != 0:
        raise SystemExit(
            "--size must be divisible by 64 "
            "(pixel_unshuffle 8 + stride 2 + two hyper stride 2)")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log = Logger(args.out)
    log("DCVC-UF shape trace")
    log(f"torch {torch.__version__} | device {device} | "
        f"size {args.size} | qp {args.qp} | backward {args.backward}")
    log("NOTE: weights are random; bpp / loss / grad values are meaningless. "
        "Only shapes and data flow matter.")
    torch.manual_seed(0)

    i_net = None
    if args.model in ("both", "intra"):
        i_net = trace_intra(args, log, device)
    if args.model in ("both", "video"):
        trace_video(args, log, device, i_net=i_net)

    log("\nDone.")
    log.save()


if __name__ == "__main__":
    main()
