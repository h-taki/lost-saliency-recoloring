"""Dinov3 module."""
from __future__ import annotations

from typing import Literal

import lightning as L
import torch

from sm_cvd.losses import MSINetLoss
from sm_cvd.metrics import cc, kld, nss as nss_metric, sim
from sm_cvd.models.dinov3_saliency import DINOv3Saliency


class DINOv3SaliencyLightning(L.LightningModule):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        repo_dir: str,
        weights: str,
        entrypoint: str = "dinov3_vitb16",
        learning_rate: float = 1e-4,
        loss_variant: Literal["method1", "method3"] = "method3",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        proc_hw: tuple[int, int] = (288, 384),
        pretrained_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = DINOv3Saliency(
            repo_dir=repo_dir, weights=weights, entrypoint=entrypoint,
            lora_rank=lora_rank, lora_alpha=lora_alpha, proc_hw=tuple(proc_hw),
        )
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and not all(
                isinstance(v, torch.Tensor) for v in state.values()
            ):
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[DINOv3SaliencyLightning] loaded {pretrained_path}, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")

        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[DINOv3SaliencyLightning] trainable {n_train/1e6:.2f}M / "
              f"total {n_total/1e6:.1f}M (rank={lora_rank}, proc_hw={tuple(proc_hw)})")

        self.criterion = MSINetLoss(variant=loss_variant)
        self._val_metrics: dict[str, list[torch.Tensor]] = {
            "cc": [], "sim": [], "kld": [], "nss": [],
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        y_hat = self(batch["image"])
        loss = self.criterion(y_hat, batch["saliency"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch["image"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        image, y_sal, y_fix = batch["image"], batch["saliency"], batch["fixation"]
        y_hat = self(image)
        loss = self.criterion(y_hat, y_sal)
        m_cc, m_sim = cc(y_sal, y_hat), sim(y_sal, y_hat)
        m_kld, m_nss = kld(y_sal, y_hat), nss_metric(y_fix, y_hat)
        self.log("val/loss", loss, prog_bar=True, batch_size=1)
        self.log("val/cc", m_cc, prog_bar=True, batch_size=1)
        self.log("val/sim", m_sim, batch_size=1)
        self.log("val/kld", m_kld, batch_size=1)
        self.log("val/nss", m_nss, batch_size=1)
        for k, v in (("cc", m_cc), ("sim", m_sim), ("kld", m_kld), ("nss", m_nss)):
            self._val_metrics[k].append(v.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        for k, vs in self._val_metrics.items():
            if vs:
                self.log(f"val/{k}_epoch", torch.stack(vs).mean())
        for k in self._val_metrics:
            self._val_metrics[k].clear()

    def configure_optimizers(self):
        return torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.hparams.learning_rate,
        )


__all__ = ["DINOv3SaliencyLightning"]
