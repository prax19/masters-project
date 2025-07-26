import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from utils.encoding import ClassMapper

import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")

import psutil

class WeightedLoss(nn.Module):
    """Opakowuje dowolną funkcję straty i wagę, zwraca weight * loss_fn(...)"""
    def __init__(self, loss_fn: nn.Module, weight: float):
        super().__init__()
        self.loss_fn = loss_fn
        self.weight = weight

    def forward(self, logits, target):
        return self.weight * self.loss_fn(logits, target)

class ExperimentalModel(pl.LightningModule):

    def __init__(
        self,
        model_backbone: nn.Module,
        encoder: str,
        weights: str,
        mapper: ClassMapper,
        losses: dict[str, WeightedLoss] | None = None,
        in_channels=3,
        t_max: int = 37150,
        lr=2e-4,
        **kwargs
    ):
        super().__init__()
        self.psutil = psutil.Process()
        self.mapper = mapper
        self.t_max = t_max
        self.save_hyperparameters()

        self.model = model_backbone(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=len(self.mapper.mapping),
            **kwargs
        )

        params = smp.encoders.get_preprocessing_params(encoder)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        if losses is None:
            losses = {
                'dice':  WeightedLoss(
                    smp.losses.DiceLoss(
                        smp.losses.MULTICLASS_MODE,
                        from_logits=True,
                        ignore_index=255
                    ), 0.8
                ),
                'focal': WeightedLoss(
                    smp.losses.FocalLoss(
                        smp.losses.MULTICLASS_MODE,
                        alpha=0.25, gamma=2.0,
                        ignore_index=255
                    ), 0.2
                ),
            }
        self.losses = nn.ModuleDict(losses)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        return self.model(image)
    
    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        mask = mask.long()

        if mask.ndim == 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)    # teraz [B,H,W]

        assert mask.ndim == 3, f"Oczekiwano mask.ndim==3, a mamy {mask.ndim}"

        # Predict mask logits
        logits_mask = self.forward(image).contiguous()

        assert (
            logits_mask.shape[1] == len(self.mapper.mapping)
        )  # [batch_size, len(self.mapper), H, W]

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = sum(loss_module(logits_mask, mask) for loss_module in self.losses.values())
        pred_mask = logits_mask.softmax(dim=1).argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", 
            num_classes=len(self.mapper.mapping), ignore_index=255
        )

        to_logger = (stage == "val")
        self.log(name=f"{stage}_loss", value=float(loss), prog_bar=True, logger=to_logger, on_step=False, on_epoch=True)
        self.log(name=f"{stage}_memory_gpu", value=(torch.cuda.max_memory_allocated() / (1024**3)), prog_bar=True, logger=to_logger, on_step=False, on_epoch=True)
        self.log(name=f"{stage}_memory_cpu", value=(self.psutil.memory_info().rss / (1024**3)), prog_bar=True, logger=to_logger, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
    def shared_epoch_end(self, outputs, stage: str):
        if not outputs:
            return
        tp = torch.cat([o['tp'] for o in outputs])
        fp = torch.cat([o['fp'] for o in outputs])
        fn = torch.cat([o['fn'] for o in outputs])
        tn = torch.cat([o['tn'] for o in outputs])

        # common metrics
        miou_dataset = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        miou_image = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        prec = smp.metrics.precision(tp, fp, fn, tn, reduction="macro")
        rec = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

        # learning rate always prog_bar
        lr = self.optimizers().param_groups[0]['lr']
        main_metrics = {f"{stage}_miou_dataset": miou_dataset, f"{stage}_miou_image": miou_image, "lr": lr}
        extra_metrics = {f"{stage}_acc_macro": acc, f"{stage}_prec_macro": prec,
                         f"{stage}_rec_macro": rec, f"{stage}_f1_macro": f1}

        if stage == "val":
            # per-class
            per_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
            if per_iou.ndim == 2: per_iou = per_iou.mean(0)
            per_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
            if per_f1.ndim == 2: per_f1 = per_f1.mean(0)
            class_metrics = {}
            for idx, (iou_v, f1_v) in enumerate(zip(per_iou, per_f1)):
                _, name, _ = self.mapper.mapping.get(idx, (idx, f"class_{idx}", None))
                class_metrics[f"iou_{name}"] = iou_v.item()
                class_metrics[f"f1_{name}"] = f1_v.item()
            # log all val metrics
            self.log_dict(main_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            self.log_dict(extra_metrics, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            if class_metrics:
                self.log_dict(class_metrics, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        else:
            # only show train metrics in progress bar, no CSV logging
            self.log_dict(main_metrics, prog_bar=True, logger=False, on_step=False, on_epoch=True)
            self.log_dict(extra_metrics, prog_bar=False, logger=False, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "val")
        self.validation_step_outputs.append(out)
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