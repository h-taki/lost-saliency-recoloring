"""Convert external checkpoints to the local state-dict format."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_SALICON_GDRIVE_ID = "1Th7fqVYx25ePMZz4LYsjNQWgAu8tJqwL"  # model_salicon_gpu


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def ensure_pb(work_dir: Path, gdrive_id: str) -> Path:
    """(documentation omitted in the English release)."""
    work_dir.mkdir(parents=True, exist_ok=True)


    existing = list(work_dir.glob("*.pb"))
    if existing:
        print(f"[ok] using cached .pb: {existing[0]}")
        return existing[0]

    try:
        import gdown
    except ImportError as e:
        raise SystemExit(
            "gdown is required: pip install gdown"
        ) from e

    zip_path = work_dir / "salicon_weights.zip"
    if not zip_path.exists():
        url = f"https://drive.google.com/uc?id={gdrive_id}&export=download"
        print(f">> downloading from {url} -> {zip_path}", flush=True)
        gdown.download(url, str(zip_path), quiet=False)

    print(f">> extracting {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work_dir)

    pbs = list(work_dir.glob("*.pb"))
    if not pbs:
        raise RuntimeError(f"No .pb file found after extraction in {work_dir}")
    print(f"[ok] extracted: {pbs[0]}")
    return pbs[0]


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def load_const_tensors(pb_path: Path) -> dict[str, "np.ndarray"]:
    """(documentation omitted in the English release)."""
    import numpy as np


    try:
        import tensorflow as tf
        graph_def = tf.compat.v1.GraphDef()
    except ImportError as e:

        raise SystemExit("tensorflow is required") from e

    with open(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    print(f">> graph has {len(graph_def.node)} nodes", flush=True)

    consts: dict[str, np.ndarray] = {}
    for node in graph_def.node:
        if node.op != "Const":
            continue
        try:
            tensor_proto = node.attr["value"].tensor
            arr = tf.make_ndarray(tensor_proto)
        except Exception as e:

            print(f"  skip {node.name}: {e}")
            continue
        consts[node.name] = arr
    print(f">> extracted {len(consts)} const tensors", flush=True)
    return consts


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------

NAME_MAP: dict[str, str] = {
    # Encoder (VGG16)
    "conv1/conv1_1": "encoder.conv1_1",
    "conv1/conv1_2": "encoder.conv1_2",
    "conv2/conv2_1": "encoder.conv2_1",
    "conv2/conv2_2": "encoder.conv2_2",
    "conv3/conv3_1": "encoder.conv3_1",
    "conv3/conv3_2": "encoder.conv3_2",
    "conv3/conv3_3": "encoder.conv3_3",
    "conv4/conv4_1": "encoder.conv4_1",
    "conv4/conv4_2": "encoder.conv4_2",
    "conv4/conv4_3": "encoder.conv4_3",
    "conv5/conv5_1": "encoder.conv5_1",
    "conv5/conv5_2": "encoder.conv5_2",
    "conv5/conv5_3": "encoder.conv5_3",
    # ASPP
    "aspp/conv1_1": "aspp.branch1",
    "aspp/conv1_2": "aspp.branch2",
    "aspp/conv1_3": "aspp.branch3",
    "aspp/conv1_4": "aspp.branch4",
    "aspp/conv1_5": "aspp.branch5_conv",
    "aspp/conv2": "aspp.fuse",
    # Decoder
    "decoder/conv1": "decoder.conv1",
    "decoder/conv2": "decoder.conv2",
    "decoder/conv3": "decoder.conv3",
    "decoder/conv4": "decoder.conv4",
}


def build_pytorch_state(consts: dict, verbose: bool = True) -> dict:
    """(documentation omitted in the English release)."""
    import numpy as np
    import torch

    state: dict[str, torch.Tensor] = {}
    missing: list[str] = []

    for tf_layer, pt_module in NAME_MAP.items():
        kernel_key_candidates = [f"{tf_layer}/kernel", f"{tf_layer}/weights"]
        bias_key_candidates = [f"{tf_layer}/bias", f"{tf_layer}/biases"]

        kernel = next((consts[k] for k in kernel_key_candidates if k in consts), None)
        bias = next((consts[k] for k in bias_key_candidates if k in consts), None)

        if kernel is None or bias is None:
            missing.append(tf_layer)
            if verbose:
                print(f"[miss] {tf_layer} (kernel={kernel is not None}, bias={bias is not None})")
            continue

        # [H, W, in, out] -> [out, in, H, W]
        if kernel.ndim != 4:
            raise RuntimeError(f"Unexpected kernel ndim for {tf_layer}: {kernel.shape}")
        kernel_t = torch.from_numpy(kernel.transpose(3, 2, 0, 1).copy()).float()
        bias_t = torch.from_numpy(bias.copy()).float()
        state[f"{pt_module}.weight"] = kernel_t
        state[f"{pt_module}.bias"] = bias_t
        if verbose:
            print(f"[ok]   {tf_layer:30s} -> {pt_module}: weight={tuple(kernel_t.shape)}, bias={tuple(bias_t.shape)}")

    if missing:
        print(f"\n[warn] {len(missing)} layers had no const tensors: {missing}")

    return state


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("weights/msinet_salicon_pytorch.pth"),
                        help="output path for the converted PyTorch state_dict")
    parser.add_argument("--work-dir", type=Path,
                        default=Path("weights/_salicon_pb"),
                        help="download/extract location for the .pb")
    parser.add_argument("--gdrive-id", type=str, default=DEFAULT_SALICON_GDRIVE_ID,
                        help="Google Drive ID of model_salicon_gpu")
    parser.add_argument("--dump-only", action="store_true",
                        help="print Const node names/shapes and exit")
    args = parser.parse_args()

    pb_path = ensure_pb(args.work_dir, args.gdrive_id)
    consts = load_const_tensors(pb_path)

    print("\n=== Const tensors (name, shape) ===")
    for name, arr in sorted(consts.items()):
        print(f"  {name:60s} {arr.shape}")

    if args.dump_only:
        return

    print("\n=== Building PyTorch state_dict ===")
    state = build_pytorch_state(consts)

    if not state:
        raise SystemExit("conversion result is empty; please check the Const node names.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(state, args.output)
    print(f"\n>> saved {len(state)} tensors to {args.output}")
    print("\n[hint] point `pretrained_path:` in configs/msi_d_hf.yaml to this file.")


if __name__ == "__main__":
    main()
