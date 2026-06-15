"""Tier 3 J: Lightning module for MSI-Net + configurable VGG depth."""
from __future__ import annotations

from typing import Literal

import lightning as L
import torch

from sm_cvd.losses import MSINetLoss
from sm_cvd.metrics import cc, kld, nss, sim
from sm_cvd.models.vgg_variants import MSINetVGGVariant, load_torchvision_vgg_weights


class MSINetVGGVariantLightning(L.LightningModule):
    """MSI-Net architecture with configurable VGG depth (11/13/16/19)."""

    def __init__(
        self,
        vgg_variant: Literal["vgg11", "vgg13", "vgg16", "vgg19"] = "vgg16",
        learning_rate: float = 1e-5,
        loss_variant: Literal["method1", "method3"] = "method3",
        pretrained_path: str | None = None,
        pretrained_imagenet: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = MSINetVGGVariant(variant=vgg_variant, subtract_imagenet_mean=True)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and not all(
                isinstance(v, torch.Tensor) for v in state.values()
            ):
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[MSINetVGGVariantLightning/{vgg_variant}] loaded pretrained_path={pretrained_path}, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")
        elif pretrained_imagenet:
            n_loaded = load_torchvision_vgg_weights(self.model)
            print(f"[MSINetVGGVariantLightning/{vgg_variant}] ImageNet VGG weights loaded: "
                  f"{n_loaded} conv layers")

        self.criterion = MSINetLoss(variant=loss_variant)
        self._val_metrics: dict[str, list[torch.Tensor]] = {
            "cc": [], "sim": [], "kld": [], "nss": [],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        pred = self.model(batch["image"])
        loss = self.criterion(pred, batch["saliency"])
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["image"].size(0))
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        image = batch["image"]
        saliency = batch["saliency"]
        fixation = batch.get("fixation")
        pred = self.model(image)
        loss = self.criterion(pred, saliency)
        val_cc = cc(saliency, pred)
        val_sim = sim(saliency, pred)
        val_kld = kld(saliency, pred)
        val_nss = nss(fixation, pred) if fixation is not None else torch.tensor(0.0)
        self.log("val/loss", loss, prog_bar=True, batch_size=1)
        self.log("val/cc", val_cc, prog_bar=True, batch_size=1)
        for k, v in {"cc": val_cc, "sim": val_sim, "kld": val_kld, "nss": val_nss}.items():
            self._val_metrics[k].append(v.detach())
        return {"loss": loss, "cc": val_cc}

    def on_validation_epoch_end(self) -> None:
        for k, vals in self._val_metrics.items():
            if vals:
                self.log(f"val/{k}_epoch", torch.stack(vals).mean())
                vals.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


__all__ = ["MSINetVGGVariantLightning"]
