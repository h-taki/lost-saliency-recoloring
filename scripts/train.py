"""Training entry point for the saliency model."""
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

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from sm_cvd.callbacks import ExperimentMetaWriter, MetricsCsvWriter



_AUTO_CALLBACK_CLASSES = (MetricsCsvWriter, ExperimentMetaWriter)


def _make_best_ckpt_callback() -> ModelCheckpoint:
    """(documentation omitted in the English release)."""
    return ModelCheckpoint(
        monitor="val/cc_epoch",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}",
        save_last=True,
        auto_insert_metric_name=False,
    )


class SmCvdLightningCLI(LightningCLI):
    """(documentation omitted in the English release)."""

    def instantiate_trainer(self, **kwargs):  # type: ignore[override]
        callbacks = list(kwargs.get("callbacks") or [])


        callbacks = [
            cb for cb in callbacks
            if not (isinstance(cb, ModelCheckpoint) and cb.monitor is None)
        ]


        has_best_ckpt = any(
            isinstance(cb, ModelCheckpoint) and cb.monitor == "val/cc_epoch"
            for cb in callbacks
        )
        if not has_best_ckpt:
            callbacks.append(_make_best_ckpt_callback())


        existing_classes = {type(cb) for cb in callbacks}
        for cls in _AUTO_CALLBACK_CLASSES:
            if cls not in existing_classes:
                callbacks.append(cls())


        print(
            "[SmCvdLightningCLI] active callbacks:",
            ", ".join(type(cb).__name__ + (f"(monitor={cb.monitor})" if isinstance(cb, ModelCheckpoint) else "")
                      for cb in callbacks),
        )

        kwargs["callbacks"] = callbacks
        return super().instantiate_trainer(**kwargs)


def main() -> None:
    SmCvdLightningCLI(
        save_config_kwargs={"overwrite": True},
    )


    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
