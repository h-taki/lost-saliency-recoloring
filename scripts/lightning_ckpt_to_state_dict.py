"""Convert a PyTorch Lightning checkpoint to a plain state dict."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, required=True,
                        help="Lightning ckpt (.ckpt) path")
    parser.add_argument("--out", dest="out_path", type=Path, required=True,
                        help="output .pth (state_dict after stripping the model.* prefix)")
    parser.add_argument("--prefix", default="model.",
                        help="prefix to strip (default: 'model.')")
    parser.add_argument("--keep-others", action="store_true",
                        help="also save keys outside the prefix (e.g. criterion.*)")
    args = parser.parse_args()

    print(f"in : {args.in_path}")
    ckpt = torch.load(args.in_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"'state_dict' key not found: {args.in_path}\n"
            "Please specify a Lightning-format .ckpt."
        )
    sd = ckpt["state_dict"]
    total = len(sd)

    new_sd: dict[str, torch.Tensor] = {}
    n_stripped = 0
    n_skipped = 0
    for k, v in sd.items():
        if k.startswith(args.prefix):
            new_sd[k[len(args.prefix):]] = v
            n_stripped += 1
        elif args.keep_others:
            new_sd[k] = v
        else:
            n_skipped += 1

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_sd, args.out_path)

    print(f"out: {args.out_path}")
    print(f"total keys in ckpt:    {total}")
    print(f"keys with '{args.prefix}' prefix (saved): {n_stripped}")
    print(f"other keys (skipped):  {n_skipped}")
    if new_sd:
        sample_keys = list(new_sd.keys())[:3]
        print(f"sample keys: {sample_keys}")


if __name__ == "__main__":
    main()
