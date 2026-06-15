"""Vgg plain module."""
from __future__ import annotations

from typing import Literal

import lightning as L
import torch

from sm_cvd.losses import MSINetLoss
from sm_cvd.metrics import cc, kld, nss, sim
from sm_cvd.models import VGGPlain


class VGGPlainLightning(L.LightningModule):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        learning_rate: float = 1e-5,
        loss_variant: Literal["method1", "method3"] = "method3",
        pretrained_vgg16: bool = True,
        pretrained_path: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = VGGPlain(subtract_imagenet_mean=True)

        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and not all(
                isinstance(v, torch.Tensor) for v in state.values()
            ):
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(
                f"[VGGPlainLightning] Loaded pretrained_path={pretrained_path}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        elif pretrained_vgg16:
            self.model.load_vgg16_imagenet()

        if freeze_encoder:
            n_frozen = 0
            for p in self.model.encoder.parameters():
                p.requires_grad = False
                n_frozen += 1
            print(f"[VGGPlainLightning] encoder frozen ({n_frozen} parameter tensors)")

        self.criterion = MSINetLoss(variant=loss_variant)

        self._val_metrics: dict[str, list[torch.Tensor]] = {
            "cc": [], "sim": [], "kld": [], "nss": [],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        y_hat = self(batch["image"])
        loss = self.criterion(y_hat, batch["saliency"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch["image"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        y_hat = self(batch["image"])
        y_sal = batch["saliency"]
        y_fix = batch["fixation"]
        loss = self.criterion(y_hat, y_sal)

        m_cc = cc(y_sal, y_hat)
        m_sim = sim(y_sal, y_hat)
        m_kld = kld(y_sal, y_hat)
        m_nss = nss(y_fix, y_hat)

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

    def test_step(self, batch, batch_idx: int) -> dict:
        y_hat = self(batch["image"])
        y_sal = batch["saliency"]
        y_fix = batch["fixation"]
        result = {
            "cc": cc(y_sal, y_hat).detach().cpu(),
            "sim": sim(y_sal, y_hat).detach().cpu(),
            "kld": kld(y_sal, y_hat).detach().cpu(),
            "nss": nss(y_fix, y_hat).detach().cpu(),
            "image_path": batch["image_path"][0],
        }
        self.log("test/cc", result["cc"], batch_size=1)
        self.log("test/sim", result["sim"], batch_size=1)
        self.log("test/kld", result["kld"], batch_size=1)
        self.log("test/nss", result["nss"], batch_size=1)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


__all__ = ["VGGPlainLightning"]
