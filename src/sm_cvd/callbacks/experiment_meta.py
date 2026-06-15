"""Callback that records experiment metadata."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml
from lightning.pytorch import Callback


def _safe_run(cmd: List[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_info(repo: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "commit": None, "commit_short": None, "branch": None,
        "dirty": None, "dirty_files": None,
    }
    commit = _safe_run(["git", "rev-parse", "HEAD"], repo)
    if commit is None:
        return info
    info["commit"] = commit
    info["commit_short"] = commit[:7]
    info["branch"] = _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
    porcelain = _safe_run(["git", "status", "--porcelain"], repo)
    if porcelain is None:
        return info
    files: List[str] = []
    for line in porcelain.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            files.append(parts[1])
        elif len(parts) == 1:
            files.append(parts[0])
    info["dirty"] = bool(files)
    info["dirty_files"] = files if files else None
    return info


def _library_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            try:
                out["gpu"] = torch.cuda.get_device_name(0)
            except Exception:
                out["gpu"] = None
        else:
            out["gpu"] = None
    except Exception:
        out["torch"] = None
        out["cuda"] = None
        out["gpu"] = None
    try:
        import lightning
        out["lightning"] = lightning.__version__
    except Exception:
        out["lightning"] = None
    return out


def _to_yaml_safe(obj: Any) -> Any:
    """(documentation omitted in the English release)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)
    if hasattr(obj, "value") and not callable(obj.value):
        return _to_yaml_safe(obj.value)
    return str(obj)


class ExperimentMetaWriter(Callback):
    """(documentation omitted in the English release)."""

    META_FILENAME = "experiment_meta.yaml"

    def __init__(self) -> None:
        super().__init__()
        self._meta: Dict[str, Any] = {}
        self._started_at: datetime | None = None
        self._wrote_initial: bool = False

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module) -> None:
        """(documentation omitted in the English release)."""
        self._write_initial_if_needed(trainer, stage="fit")

    def on_validate_start(self, trainer, pl_module) -> None:
        self._write_initial_if_needed(trainer, stage="validate")

    def on_test_start(self, trainer, pl_module) -> None:
        self._write_initial_if_needed(trainer, stage="test")

    def on_predict_start(self, trainer, pl_module) -> None:
        self._write_initial_if_needed(trainer, stage="predict")

    def teardown(self, trainer, pl_module, stage) -> None:
        if not self._wrote_initial:


            self._write_initial_if_needed(trainer, stage=getattr(stage, "value", str(stage)))
            if not self._wrote_initial:
                return
        if self._meta.get("status") in ("crashed", "interrupted"):
            self._fill_end_time()
            self._flush(trainer)
            return
        self._meta["status"] = "completed"
        self._fill_end_time()
        self._flush(trainer)

    def on_exception(self, trainer, pl_module, exception) -> None:
        if not self._wrote_initial:
            return
        if isinstance(exception, KeyboardInterrupt):
            self._meta["status"] = "interrupted"
        else:
            self._meta["status"] = "crashed"
        self._meta["exception_type"] = type(exception).__name__
        self._meta["exception_msg"] = str(exception)[:500]
        self._fill_end_time()
        self._flush(trainer)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _write_initial_if_needed(self, trainer, stage: str) -> None:
        if self._wrote_initial:
            return
        log_dir = getattr(trainer, "log_dir", None)
        if log_dir is None:


            return
        self._started_at = datetime.now(timezone.utc).astimezone()
        self._meta = self._collect_initial_meta(trainer, stage)
        self._meta["status"] = "running"
        self._meta["started_at"] = self._started_at.isoformat(timespec="seconds")
        self._meta["ended_at"] = None
        self._meta["elapsed_sec"] = None
        self._flush(trainer)
        self._wrote_initial = True

    def _collect_initial_meta(self, trainer, stage: str) -> Dict[str, Any]:
        repo = Path(__file__).resolve().parents[3]
        default_root = Path(getattr(trainer, "default_root_dir", ".") or ".")
        try:
            exp_dir = default_root.parent.name
        except Exception:
            exp_dir = None
        fold: int | None = None
        try:
            tail = default_root.name
            if tail.startswith("fold_"):
                fold = int(tail.split("_", 1)[1])
        except Exception:
            fold = None
        log_dir = Path(trainer.log_dir) if trainer.log_dir else default_root

        meta: Dict[str, Any] = {
            "exp_name": exp_dir,
            "fold": fold,
            "default_root_dir": str(default_root).replace("\\", "/"),
            "log_dir": str(log_dir).replace("\\", "/"),
            "stage": getattr(stage, "value", str(stage)),
            "git": _git_info(repo),
            "config": {
                "source_path": self._infer_source_config_path(),
                "resolved_path": "./config.yaml",
            },
            "command_line": [str(x) for x in sys.argv],
            "cwd": os.getcwd().replace("\\", "/"),
            "python": sys.version.split()[0],
            "hostname": socket.gethostname(),
            "versions": _library_versions(),
        }
        return meta

    @staticmethod
    def _infer_source_config_path() -> str | None:
        argv = sys.argv
        for i, tok in enumerate(argv):
            if tok == "--config" and i + 1 < len(argv):
                return argv[i + 1]
            if tok.startswith("--config="):
                return tok.split("=", 1)[1]
        return None

    def _fill_end_time(self) -> None:
        now = datetime.now(timezone.utc).astimezone()
        self._meta["ended_at"] = now.isoformat(timespec="seconds")
        if self._started_at is not None:
            self._meta["elapsed_sec"] = int(
                (now - self._started_at).total_seconds()
            )

    def _flush(self, trainer) -> None:
        log_dir = getattr(trainer, "log_dir", None)
        if log_dir is None:
            return
        path = Path(log_dir) / self.META_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_meta = _to_yaml_safe(self._meta)
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(safe_meta, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        except Exception as exc:
            sys.stderr.write(f"[ExperimentMetaWriter] failed to write {path}: {exc}\n")


__all__ = ["ExperimentMetaWriter"]
