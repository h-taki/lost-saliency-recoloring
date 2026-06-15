"""PyTorch Lightning module wrapping the saliency model, loss, and metrics."""
from __future__ import annotations

from typing import Literal

import lightning as L
import torch
from torch import nn

from sm_cvd.losses import MSINetLoss
from sm_cvd.metrics import cc, kld, nss, sim
from sm_cvd.models import MSINet


class MSINetLightning(L.LightningModule):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        learning_rate: float = 1e-5,
        loss_variant: Literal["method1", "method3"] = "method1",
        pretrained_vgg16: bool = True,
        pretrained_path: str | None = None,
        freeze_encoder: bool = False,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        kd_weight: float = 0.0,
        recon_weight: float = 0.0,
        optimizer_name: Literal["adam", "adamw"] = "adam",
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = MSINet(subtract_imagenet_mean=True)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")

            if isinstance(state, dict) and "state_dict" in state and not all(
                isinstance(v, torch.Tensor) for v in state.values()
            ):
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)

            print(
                f"[MSINetLightning] Loaded pretrained_path={pretrained_path}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        elif pretrained_vgg16:
            self.model.load_vgg16_imagenet()

        if freeze_encoder:
            n_frozen = 0
            for p in self.model.encoder.parameters():
                p.requires_grad = False
                n_frozen += 1
            print(f"[MSINetLightning] encoder frozen ({n_frozen} parameter tensors)")




        self.recon_head = None
        if recon_weight > 0.0:
            from torch.nn import Conv2d, ReLU, Sequential, Upsample
            self.recon_head = Sequential(
                Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Conv2d(256, 128, 3, padding=1), ReLU(inplace=True),
                Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Conv2d(128, 64, 3, padding=1), ReLU(inplace=True),
                Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Conv2d(64, 32, 3, padding=1), ReLU(inplace=True),
                Conv2d(32, 3, 1),
            )
            print(f"[MSINetLightning] reconstruction head added "
                  f"(recon_weight={recon_weight})")

        self.criterion = MSINetLoss(variant=loss_variant)


        self._val_metrics: dict[str, list[torch.Tensor]] = {
            "cc": [], "sim": [], "kld": [], "nss": [],
        }

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _apply_mixup_cutmix(
        self, image: torch.Tensor, saliency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(documentation omitted in the English release)."""
        mixup_a = float(self.hparams.mixup_alpha)
        cutmix_a = float(self.hparams.cutmix_alpha)
        if mixup_a <= 0.0 and cutmix_a <= 0.0:
            return image, saliency

        B = image.shape[0]
        if B < 2:
            return image, saliency


        perm = torch.randperm(B, device=image.device)


        use_mixup = mixup_a > 0.0
        use_cutmix = cutmix_a > 0.0
        if use_mixup and use_cutmix:
            use_mixup = torch.rand(1).item() < 0.5
            use_cutmix = not use_mixup

        if use_mixup:
            lam = float(torch.distributions.Beta(mixup_a, mixup_a).sample().item())
            image = lam * image + (1.0 - lam) * image[perm]
            saliency = lam * saliency + (1.0 - lam) * saliency[perm]
        elif use_cutmix:
            lam = float(torch.distributions.Beta(cutmix_a, cutmix_a).sample().item())
            _, _, H, W = image.shape
            cut_ratio = (1.0 - lam) ** 0.5
            cut_h = int(H * cut_ratio)
            cut_w = int(W * cut_ratio)
            cy = torch.randint(0, H, (1,)).item()
            cx = torch.randint(0, W, (1,)).item()
            y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
            x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
            image[:, :, y1:y2, x1:x2] = image[perm][:, :, y1:y2, x1:x2]
            saliency[:, :, y1:y2, x1:x2] = saliency[perm][:, :, y1:y2, x1:x2]

        return image, saliency

    @staticmethod
    def _kd_kl_loss(student_pred: torch.Tensor, teacher_sal: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
        """(documentation omitted in the English release)."""
        B = student_pred.shape[0]
        s = student_pred.reshape(B, -1).clamp(min=eps)
        t = teacher_sal.reshape(B, -1).clamp(min=eps)
        s = s / s.sum(dim=1, keepdim=True)
        t = t / t.sum(dim=1, keepdim=True)
        # KL(t || s) = sum_t t * log(t / s)
        kl = (t * (t.log() - s.log())).sum(dim=1).mean()
        return kl

    def _forward_with_recon(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(documentation omitted in the English release)."""
        from sm_cvd.models.msinet import normalize_minmax
        m = self.model
        x = image - m._mean if m.subtract_imagenet_mean else image
        feats = m.encoder(x)
        ctx = m.aspp(feats)                              # (B, 256, H/8, W/8)
        saliency = m.decoder(ctx)
        saliency = normalize_minmax(saliency)
        recon = self.recon_head(ctx) if self.recon_head is not None else None
        return saliency, recon

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        import torch.nn.functional as F
        image = batch["image"]
        saliency = batch["saliency"]
        image, saliency = self._apply_mixup_cutmix(image, saliency)

        recon_w = float(self.hparams.recon_weight)
        if recon_w > 0.0:
            y_hat, recon = self._forward_with_recon(image)
        else:
            y_hat = self(image)
            recon = None

        loss = self.criterion(y_hat, saliency)


        kd_w = float(self.hparams.kd_weight)
        if kd_w > 0.0 and "teacher_saliency" in batch:
            kd_loss = self._kd_kl_loss(y_hat, batch["teacher_saliency"])
            loss = loss + kd_w * kd_loss
            self.log("train/kd_loss", kd_loss, on_step=True, on_epoch=True,
                     prog_bar=False, batch_size=image.shape[0])


        if recon is not None:
            recon_target = image / 255.0
            recon_loss = F.mse_loss(recon, recon_target)
            loss = loss + recon_w * recon_loss
            self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True,
                     prog_bar=False, batch_size=image.shape[0])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=image.shape[0])
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

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def configure_optimizers(self):

        opt_name = getattr(self.hparams, "optimizer_name", "adam")
        wd = getattr(self.hparams, "weight_decay", 0.0)
        if opt_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=wd)
        # adam (default)
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=wd)


__all__ = ["MSINetLightning"]
