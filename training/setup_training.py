import os
import yaml
import pandas as pd
import datetime

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.tuner import Tuner

from segmentation_models_pytorch.base import SegmentationModel
from torch.utils.data import DataLoader

from utils.encoding import ClassMapper
from utils.benchmark import start_benchmark
from training.model import ExperimentalModel

def setup_training(
    train_loader: DataLoader,
    val_loader: DataLoader,
    mapper: ClassMapper,
    backbone: SegmentationModel,
    encoder: str,
    weights: str,
    losses: dict,
    epochs: int = 50,
):
    EPOCHS = epochs
    T_MAX = EPOCHS * len(train_loader)

    model = ExperimentalModel(
        model_backbone=backbone,
        weights=weights,
        encoder=encoder,
        in_channels=3,
        t_max=T_MAX,
        losses=losses,
        mapper=mapper
    )

    logger = CSVLogger("logs", name=backbone.__name__)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        precision="16-mixed",
        logger=logger,
        val_check_interval=743/len(train_loader),
        max_steps=T_MAX
    )

    # Strojenie learning rate
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        min_lr=1e-6,
        max_lr=1e-2,
        num_training=743,
        early_stop_threshold=None
    )

    new_lr = lr_finder.suggestion()

    model.hparams.lr = new_lr

    # Trening
    training_start_time = datetime.datetime.now()
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    training_finish_time = datetime.datetime.now()

    # Naprawa pliku z metrykami
    log_path = trainer.logger.log_dir
    # log_file_path = os.path.join(".", log_path, "metrics.csv")

    # log = pd.read_csv(log_file_path, sep=',')
    # log = log.groupby('step').mean()
    # log.index.name = "step"

    # log.to_csv(os.path.join(".", log_path, f"metrics_fixed.csv"))

    latency, fps, aloc, rese, peak = start_benchmark(model)

    flat_losses = {}
    for name, wrapper in losses.items():
        flat_losses[name] = wrapper.weight

    # metadata
    metadata = f"""
        'model':
        - 'name': {backbone.__name__}
        - 'encoder': {encoder}
        - 'weights': {weights}
        - 'loss_fn': {flat_losses}
        - 'peak_lr': {new_lr}
        - 'time':
            - 'start': {training_start_time}
            - 'finish': {training_finish_time}
        - 'benchmark':
            - 'latency': {latency}
            - 'fps': {fps}
            - 'aloc_mem': {aloc}
            - 'reserved_mem': {rese}
            - 'peak_mem': {peak}
        """
    metadata = yaml.safe_load(metadata)
    with open(os.path.join('.', log_path, 'metadata.yaml'), 'w') as file:
        yaml.dump(metadata, file)