"""Vgg decoder module."""
from __future__ import annotations

from typing import Literal

import lightning as L
import torch

from sm_cvd.losses import MSINetLoss
from sm_cvd.metrics import cc, kld, nss, sim
from sm_cvd.models.decoders import D3PlainNet, D4UNet


class VGGDecoderLightning(L.LightningModule):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        decoder_variant: Literal["d3_plain", "d4_unet"] = "d3_plain",
        learning_rate: float = 1e-5,
        loss_variant: Literal["method1", "method3"] = "method3",
        pretrained_path: str | None = None,
        pretrained_vgg16_imagenet: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if decoder_variant == "d3_plain":
            self.model = D3PlainNet(subtract_imagenet_mean=True)
        elif decoder_variant == "d4_unet":
            self.model = D4UNet(subtract_imagenet_mean=True)
        else:
            raise ValueError(f"Unknown decoder_variant: {decoder_variant}")

        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and not all(
                isinstance(v, torch.Tensor) for v in state.values()
            ):
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(
                f"[VGGDecoderLightning/{decoder_variant}] Loaded "
                f"pretrained_path={pretrained_path}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        elif pretrained_vgg16_imagenet:
            self._load_vgg16_imagenet()

        self.criterion = MSINetLoss(variant=loss_variant)
        self._val_metrics: dict[str, list[torch.Tensor]] = {
            "cc": [], "sim": [], "kld": [], "nss": [],
        }

    def _load_vgg16_imagenet(self) -> None:
        """(documentation omitted in the English release)."""
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"[VGGDecoderLightning] vgg16 load failed: {e}")
            return



        #   [0] conv1_1, [2] conv1_2, [5] conv2_1, [7] conv2_2,
        #   [10] conv3_1, [12] conv3_2, [14] conv3_3,
        #   [17] conv4_1, [19] conv4_2, [21] conv4_3,
        #   [24] conv5_1, [26] conv5_2, [28] conv5_3
        torch_idx = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        our_names = [
            "conv1_1", "conv1_2",
            "conv2_1", "conv2_2",
            "conv3_1", "conv3_2", "conv3_3",
            "conv4_1", "conv4_2", "conv4_3",
            "conv5_1", "conv5_2", "conv5_3",
        ]
        encoder = self.model.encoder
        loaded = 0
        for ti, name in zip(torch_idx, our_names):
            try:
                src = vgg.features[ti]
                dst = getattr(encoder, name)

                dst.weight.data.copy_(src.weight.data)
                if dst.bias is not None and src.bias is not None:
                    dst.bias.data.copy_(src.bias.data)
                loaded += 1
            except Exception as e:
                print(f"[VGGDecoderLightning] failed to copy {name}: {e}")
        print(f"[VGGDecoderLightning] VGG16 ImageNet loaded: {loaded}/{len(torch_idx)} conv layers")

    # ----------------------------------------------------------------------
    # forward / training / validation
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        image = batch["image"]
        saliency = batch["saliency"]
        pred = self.model(image)
        loss = self.criterion(pred, saliency)
        self.log("train/loss", loss, prog_bar=True, batch_size=image.size(0))
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
        self._val_metrics["cc"].append(val_cc.detach())
        self._val_metrics["sim"].append(val_sim.detach())
        self._val_metrics["kld"].append(val_kld.detach())
        self._val_metrics["nss"].append(val_nss.detach())
        return {"loss": loss, "cc": val_cc, "sim": val_sim, "kld": val_kld, "nss": val_nss}

    def on_validation_epoch_end(self) -> None:
        for key, vals in self._val_metrics.items():
            if vals:
                mean = torch.stack(vals).mean()
                self.log(f"val/{key}_epoch", mean)
                vals.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


__all__ = ["VGGDecoderLightning"]
