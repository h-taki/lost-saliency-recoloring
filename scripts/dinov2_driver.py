"""Train and evaluate the adapted dichromat saliency predictor (frozen DINOv2 + LoRA)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRETRAIN_CFG = REPO / "configs" / "dinov2_salicon_pretrain.yaml"
FT_CFG = REPO / "configs" / "dinov2_lora_d_fold1.yaml"
PRETRAIN_DIR = REPO / "runs" / "dinov2_lora_salicon_pretrain"
PTH = REPO / "weights" / "dinov2_lora_salicon_pretrain.pth"


def run(cmd: list[str], desc: str) -> int:
    print(f"\n>>> {desc}\n  $ {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(REPO)).returncode


def find_best_ckpt() -> Path | None:
    cks = sorted(PRETRAIN_DIR.glob("lightning_logs/version_*/checkpoints/best-*.ckpt"))
    if not cks:
        cks = sorted(PRETRAIN_DIR.glob("lightning_logs/version_*/checkpoints/*.ckpt"))
    return cks[-1] if cks else None


def smoke() -> int:
    import torch
    sys.path.insert(0, str(REPO / "src"))
    from sm_cvd.models.dinov2_saliency import DINOv2Saliency
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    m = DINOv2Saliency(lora_rank=8).to(dev).eval()
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"trainable {n_train/1e6:.2f}M / total {n_total/1e6:.1f}M")
    x = torch.randn(1, 3, 240, 320, device=dev)
    with torch.no_grad():
        y = m(x)
    print(f"forward OK: in {tuple(x.shape)} -> out {tuple(y.shape)}, "
          f"range [{float(y.min()):.3f}, {float(y.max()):.3f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run"])
    ap.add_argument("--folds", nargs="+", type=int, default=[1])
    ap.add_argument("--skip-pretrain", action="store_true")
    args = ap.parse_args()

    if args.mode == "smoke":
        return smoke()

    if not args.skip_pretrain and not find_best_ckpt():
        rc = run([sys.executable, "scripts/train.py", "fit", "--config", str(PRETRAIN_CFG)],
                 "pretrain: DINOv2+LoRA on SALICON real GT (5ep)")
        if rc != 0:
            return rc
    else:
        print("[pretrain] skip (existing ckpt or --skip-pretrain)")

    if not PTH.exists():
        ck = find_best_ckpt()
        if not ck:
            print("[FATAL] no pretrain ckpt")
            return 1
        rc = run([sys.executable, "scripts/lightning_ckpt_to_state_dict.py",
                  "--in", str(ck), "--out", str(PTH)], "convert ckpt -> .pth")
        if rc != 0:
            return rc
    else:
        print(f"[convert] skip (existing {PTH.name})")

    for f in args.folds:
        rc = run([sys.executable, "scripts/run_kfold.py", "--config", str(FT_CFG),
                  "--num-folds", str(f), "--start-fold", str(f)],
                 f"finetune EToCVD fold_{f}")
        if rc != 0:
            return rc
    print("\ndone. Compare against baseline fold_1 = 0.7129 / 5-fold mean = 0.7262."
          " kill criterion: fold_1 ΔCC < +0.01")
    return 0


if __name__ == "__main__":
    sys.exit(main())
