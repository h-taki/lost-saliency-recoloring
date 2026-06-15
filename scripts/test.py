"""Evaluation entry point for a trained saliency model."""
from __future__ import annotations

import os
import sys
from pathlib import Path


if sys.flags.utf8_mode == 0:
    import subprocess
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    result = subprocess.run([sys.executable, *sys.argv], env=env)
    sys.exit(result.returncode)

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lightning.pytorch.cli import LightningCLI


def main() -> None:
    LightningCLI(save_config_kwargs={"overwrite": True})
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
