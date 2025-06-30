import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils.encoding import ClassMapper

from typing_extensions import deprecated

@deprecated("An old Lightning module.")
class UnetModel(pl.LightningModule):
    def __init__(
            self, 
            model_arch: nn.Module,
            encoder_weights, 
            encoder_name, 
            in_channels, 
            mapper: ClassMapper,
            t_max: int = 50,
            lr=2e-4,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.mapper = mapper

        self.t_max = t_max
        self.model = model_arch(
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=len(self.mapper.mapping),
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=255)
        self.focal_loss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, alpha=0.25, gamma=2.0, ignore_index=255)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.temp_train_data = {}

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        mask = mask.long()

        if mask.ndim == 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)    # teraz [B,H,W]

        assert mask.ndim == 3, f"Oczekiwano mask.ndim==3, a mamy {mask.ndim}"

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == len(self.mapper.mapping)
        )  # [batch_size, len(self.mapper), H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = (
            0.8 * self.dice_loss(logits_mask, mask) +
            0.2 * self.focal_loss(logits_mask, mask)
        )

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=len(self.mapper.mapping), ignore_index=255
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage: str):
        # sanity-check
        if len(outputs) == 0:
            return

        # batch stats
        tp = torch.cat([o["tp"] for o in outputs])   #  [N,C]
        fp = torch.cat([o["fp"] for o in outputs])
        fn = torch.cat([o["fn"] for o in outputs])
        tn = torch.cat([o["tn"] for o in outputs])

        # metrics
        miou_dataset = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        miou_image = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        acc   = smp.metrics.accuracy( tp, fp, fn, tn, reduction="macro")
        prec  = smp.metrics.precision(tp, fp, fn, tn, reduction="macro")
        rec   = smp.metrics.recall(   tp, fp, fn, tn, reduction="macro")
        f1    = smp.metrics.f1_score( tp, fp, fn, tn, reduction="macro")
        # fwIoU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="weighted")
        
        self.temp_train_data.update({
                f"lr": self.hparams.lr,                
                f"{stage}_miou_dataset": miou_dataset.detach(),
                f"{stage}_miou_image": miou_image.detach(),
                f"{stage}_acc_macro":   acc.detach(),
                f"{stage}_prec_macro":  prec.detach(),
                f"{stage}_rec_macro":   rec.detach(),
                f"{stage}_f1_macro":    f1.detach(),
                # f"{stage}_fwIoU":       fwIoU.detach(),
            })

        # per-class metrics
        per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        if per_class_iou.ndim == 2:
            per_class_iou = per_class_iou.mean(dim=0)

        f1_per_class = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
        if f1_per_class.ndim == 2:
            f1_per_class = f1_per_class.mean(dim=0)

        # logging
        if stage == "val":
            for idx, (iou_val, f1_val) in enumerate(zip(per_class_iou, f1_per_class)):
                entry = self.mapper.mapping.get(idx, (idx, f"class_{idx}", None))
                _, class_name, _ = entry
                self.temp_train_data[f"iou_{class_name}"] = iou_val.item()
                self.temp_train_data[f"f1_{class_name}"]  = f1_val.item()
            
        self.log(f"{stage}_miou",
            miou_dataset,
            prog_bar=True, logger=False, on_step=False, on_epoch=True)
            
    def save_log(self):
        for name, value in self.temp_train_data.items():
            self.log(
                name=name,
                value=value,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True
            )
        self.temp_train_data.clear()

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        self.temp_train_data.update({"train_loss": out["loss"]})
        return out

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.save_log()
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "val")
        self.validation_step_outputs.append(out)
        self.temp_train_data.update({"val_loss": out["loss"]})
        return out

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.t_max,
            pct_start=0.3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # wywołuj scheduler.step() co batch
                "frequency": 1,
            },
        }