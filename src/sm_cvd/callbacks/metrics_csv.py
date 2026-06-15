"""Callback that writes per-epoch metrics to CSV."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from lightning.pytorch import Callback


class MetricsCsvWriter(Callback):
    """(documentation omitted in the English release)."""

    def __init__(self) -> None:
        super().__init__()
        # epoch -> row dict
        self._rows: Dict[int, Dict[str, Any]] = {}
        self._fieldnames: List[str] = ["epoch", "step"]

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module) -> None:  # noqa: D401
        self._record(trainer)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: D401
        self._record(trainer)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _record(self, trainer) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        ep = int(trainer.current_epoch)
        row = self._rows.get(ep, {"epoch": ep, "step": int(trainer.global_step)})

        row["step"] = int(trainer.global_step)
        for k, v in trainer.callback_metrics.items():
            try:
                row[str(k)] = float(v)
            except (TypeError, ValueError):

                continue
        self._rows[ep] = row
        self._update_fieldnames()
        self._flush(trainer)

    def _update_fieldnames(self) -> None:
        keys: set[str] = set()
        for r in self._rows.values():
            keys.update(r.keys())
        fixed = ["epoch", "step"]
        rest = sorted(keys - set(fixed))
        self._fieldnames = fixed + rest

    def _flush(self, trainer) -> None:
        log_dir = getattr(trainer, "log_dir", None)
        if log_dir is None:
            return
        path = Path(log_dir) / "metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for ep in sorted(self._rows.keys()):
                writer.writerow(self._rows[ep])


__all__ = ["MetricsCsvWriter"]
