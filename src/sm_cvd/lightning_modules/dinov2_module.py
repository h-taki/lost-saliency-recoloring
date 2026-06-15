"""Dinov2 module."""
from __future__ import annotations

from typing import Literal

import lightning as L
import torch

from sm_cvd.losses import MSINetLoss
from sm_cvd.metrics import cc, kld, nss as nss_metric, sim
from sm_cvd.models.dinov2_saliency import DINOv2Saliency


class DINOv2SaliencyLightning(L.LightningModule):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        loss_variant: Literal["method1", "method3"] = "method3",
        backbone_name: str = "dinov2_vitb14_reg",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        pretrained_path: str | None = None,
        use_film: bool = False,
        dual_pathway: bool = False,
        nss_lambda: float = 0.0,
        decoder_variant: str = "default",
        rank_lambda: float = 0.0,
        rank_grid: int = 16,
        rank_margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()





        self.use_film = use_film
        self.dual_pathway = dual_pathway
        self.model = DINOv2Saliency(
            backbone_name=backbone_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_film=use_film,
            dual_pathway=dual_pathway,
            decoder_variant=decoder_variant,
        )
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and not all(
                isinstance(v, torch.Tensor) for v in state.values()
            ):
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[DINOv2SaliencyLightning] loaded {pretrained_path}, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")

        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[DINOv2SaliencyLightning] trainable {n_train/1e6:.2f}M / "
              f"total {n_total/1e6:.1f}M params (rank={lora_rank})")

        self.criterion = MSINetLoss(variant=loss_variant)
        self._val_metrics: dict[str, list[torch.Tensor]] = {
            "cc": [], "sim": [], "kld": [], "nss": [],
        }

    def _confusion(self, batch: dict) -> torch.Tensor | None:
        if not self.use_film:
            return None
        c = batch.get("teacher_saliency")
        if c is None:
            raise KeyError(
                "use_film=True requires confusion_map in batch['teacher_saliency']. "
                "Set teacher_saliency_dir to the confusion PNG dir in EToCVDDataModule."
            )
        if c.max() > 1.5:
            c = c / 255.0
        return c

    def _image_cvd(self, batch: dict) -> torch.Tensor | None:
        if not self.dual_pathway:
            return None
        c = batch.get("teacher_saliency")
        if c is None:
            raise KeyError(
                "dual_pathway=True requires I_cvd in batch['teacher_saliency'] (3ch). "
                "Set teacher_saliency_dir=data/etocvd_d_cvd_paired_png, teacher_channels=3."
            )

        if c.max() > 1.5:
            c = c / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=c.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=c.device).view(1, 3, 1, 1)
        return (c - mean) / std

    def forward(self, image: torch.Tensor,
                confusion: torch.Tensor | None = None,
                image_cvd: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(image, confusion, image_cvd)

    def _region_means(self, m: torch.Tensor, g: int) -> torch.Tensor:
        """(documentation omitted in the English release)."""
        pooled = torch.nn.functional.adaptive_avg_pool2d(m, (g, g))  # (B,1,g,g)
        return pooled.flatten(1)                                     # (B, g*g)

    def _rank_loss(self, y_hat: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        """(documentation omitted in the English release)."""
        g, m = self.hparams.rank_grid, self.hparams.rank_margin
        rp = self._region_means(y_hat, g)        # (B, R)
        rg = self._region_means(saliency, g)
        B, R = rp.shape
        di = rp.unsqueeze(2) - rp.unsqueeze(1)
        dg = rg.unsqueeze(2) - rg.unsqueeze(1)

        thr = 0.1 * (rg.amax(1, keepdim=True) - rg.amin(1, keepdim=True)).clamp(min=1e-6)
        sign = torch.sign(dg)
        mask = (dg.abs() > thr.unsqueeze(2)).float()
        # hinge: max(0, margin - sign*di)
        hinge = torch.clamp(m - sign * di, min=0.0) * mask
        denom = mask.sum().clamp(min=1.0)
        return hinge.sum() / denom

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        image, saliency = batch["image"], batch["saliency"]
        y_hat = self(image, self._confusion(batch), self._image_cvd(batch))
        loss = self.criterion(y_hat, saliency)

        if self.hparams.nss_lambda > 0:
            if "fixation" not in batch:
                raise KeyError(
                    "nss_lambda>0 requires batch['fixation']. "
                    "Set train_return_fixation: true in EToCVDDataModule."
                )
            m_nss = nss_metric(batch["fixation"], y_hat)
            loss = loss - self.hparams.nss_lambda * m_nss
            self.log("train/nss", m_nss, on_step=False, on_epoch=True, batch_size=1)

        if self.hparams.rank_lambda > 0:
            l_rank = self._rank_loss(y_hat, saliency)
            loss = loss + self.hparams.rank_lambda * l_rank
            self.log("train/rank", l_rank, on_step=False, on_epoch=True, batch_size=1)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=image.shape[0])
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        image, y_sal, y_fix = batch["image"], batch["saliency"], batch["fixation"]
        y_hat = self(image, self._confusion(batch), self._image_cvd(batch))
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

    def test_step(self, batch, batch_idx: int) -> dict:
        image, y_sal, y_fix = batch["image"], batch["saliency"], batch["fixation"]
        y_hat = self(image, self._confusion(batch), self._image_cvd(batch))
        result = {
            "cc": cc(y_sal, y_hat).detach().cpu(),
            "sim": sim(y_sal, y_hat).detach().cpu(),
            "kld": kld(y_sal, y_hat).detach().cpu(),
            "nss": nss_metric(y_fix, y_hat).detach().cpu(),
            "image_path": batch["image_path"][0],
        }
        for k in ("cc", "sim", "kld", "nss"):
            self.log(f"test/{k}", result[k], batch_size=1)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.hparams.learning_rate,
        )


__all__ = ["DINOv2SaliencyLightning"]
