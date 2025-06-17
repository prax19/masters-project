import os
import glob
from collections import namedtuple
import random

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as v2

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

class A2D2Dataset(VisionDataset):

    A2D2DatasetClass = namedtuple(
        "A2D2DatasetClass",
        ["name", "id", "color"],
    )

    # Based on original A2D2 dataset https://www.a2d2.audi/en/dataset/
    classes = [
        A2D2DatasetClass(name='Car 1', id=0, color=(255, 0, 0)),
        A2D2DatasetClass(name='Car 2', id=1, color=(200, 0, 0)),
        A2D2DatasetClass(name='Car 3', id=2, color=(150, 0, 0)),
        A2D2DatasetClass(name='Car 4', id=3, color=(128, 0, 0)),
        A2D2DatasetClass(name='Bicycle 1', id=4, color=(182, 89, 6)),
        A2D2DatasetClass(name='Bicycle 2', id=5, color=(150, 50, 4)),
        A2D2DatasetClass(name='Bicycle 3', id=6, color=(90, 30, 1)),
        A2D2DatasetClass(name='Bicycle 4', id=7, color=(90, 30, 30)),
        A2D2DatasetClass(name='Pedestrian 1', id=8, color=(204, 153, 255)),
        A2D2DatasetClass(name='Pedestrian 2', id=9, color=(189, 73, 155)),
        A2D2DatasetClass(name='Pedestrian 3', id=10, color=(239, 89, 191)),
        A2D2DatasetClass(name='Truck 1', id=11, color=(255, 128, 0)),
        A2D2DatasetClass(name='Truck 2', id=12, color=(200, 128, 0)),
        A2D2DatasetClass(name='Truck 3', id=13, color=(150, 128, 0)),
        A2D2DatasetClass(name='Small vehicles 1', id=14, color=(0, 255, 0)),
        A2D2DatasetClass(name='Small vehicles 2', id=15, color=(0, 200, 0)),
        A2D2DatasetClass(name='Small vehicles 3', id=16, color=(0, 150, 0)),
        A2D2DatasetClass(name='Traffic signal 1', id=17, color=(0, 128, 255)),
        A2D2DatasetClass(name='Traffic signal 2', id=18, color=(30, 28, 158)),
        A2D2DatasetClass(name='Traffic signal 3', id=19, color=(60, 28, 100)),
        A2D2DatasetClass(name='Traffic sign 1', id=20, color=(0, 255, 255)),
        A2D2DatasetClass(name='Traffic sign 2', id=21, color=(30, 220, 220)),
        A2D2DatasetClass(name='Traffic sign 3', id=22, color=(60, 157, 199)),
        A2D2DatasetClass(name='Utility vehicle 1', id=23, color=(255, 255, 0)),
        A2D2DatasetClass(name='Utility vehicle 2', id=24, color=(255, 255, 200)),
        A2D2DatasetClass(name='Sidebars', id=25, color=(233, 100, 0)),
        A2D2DatasetClass(name='Speed bumper', id=26, color=(110, 110, 0)),
        A2D2DatasetClass(name='Curbstone', id=27, color=(128, 128, 0)),
        A2D2DatasetClass(name='Solid line', id=28, color=(255, 193, 37)),
        A2D2DatasetClass(name='Irrelevant signs', id=29, color=(64, 0, 64)),
        A2D2DatasetClass(name='Road blocks', id=30, color=(185, 122, 87)),
        A2D2DatasetClass(name='Tractor', id=31, color=(0, 0, 100)),
        A2D2DatasetClass(name='Non-drivable street', id=32, color=(139, 99, 108)),
        A2D2DatasetClass(name='Zebra crossing', id=33, color=(210, 50, 115)),
        A2D2DatasetClass(name='Obstacles / trash', id=34, color=(255, 0, 128)),
        A2D2DatasetClass(name='Poles', id=35, color=(255, 246, 143)),
        A2D2DatasetClass(name='RD restricted area', id=36, color=(150, 0, 150)),
        A2D2DatasetClass(name='Animals', id=37, color=(204, 255, 153)),
        A2D2DatasetClass(name='Grid structure', id=38, color=(238, 162, 173)),
        A2D2DatasetClass(name='Signal corpus', id=39, color=(33, 44, 177)),
        A2D2DatasetClass(name='Drivable cobblestone', id=40, color=(180, 50, 180)),
        A2D2DatasetClass(name='Electronic traffic', id=41, color=(255, 70, 185)),
        A2D2DatasetClass(name='Slow drive area', id=42, color=(238, 233, 191)),
        A2D2DatasetClass(name='Nature object', id=43, color=(147, 253, 194)),
        A2D2DatasetClass(name='Parking area', id=44, color=(150, 150, 200)),
        A2D2DatasetClass(name='Sidewalk', id=45, color=(180, 150, 200)),
        A2D2DatasetClass(name='Ego car', id=46, color=(72, 209, 204)),
        A2D2DatasetClass(name='Painted driv. instr.', id=47, color=(200, 125, 210)),
        A2D2DatasetClass(name='Traffic guide obj.', id=48, color=(159, 121, 238)),
        A2D2DatasetClass(name='Dashed line', id=49, color=(128, 0, 255)),
        A2D2DatasetClass(name='RD normal street', id=50, color=(255, 0, 255)),
        A2D2DatasetClass(name='Sky', id=51, color=(135, 206, 255)),
        A2D2DatasetClass(name='Buildings', id=52, color=(241, 230, 255)),
        A2D2DatasetClass(name='Blurred area', id=53, color=(96, 69, 143)),
        A2D2DatasetClass(name='Rain dirt', id=54, color=(53, 46, 82))
    ]

    def __init__(
            self,
            root: Union[str, Path],
            split: str = "train",
            scenes = None,
            camera = 'cam_front_center',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        root_semantic = os.path.join(root, 'camera_lidar_semantic')
        self.scenes = scenes or sorted(os.listdir(root_semantic))
        self.camera = camera
        self.camera_id = camera.replace('cam_', '').replace('_', '')
        self.images = []
        self.targets = []

        if split is not None:
            verify_str_arg(split, "split", ("train", "val"))
            if split=='train':
                self.scenes, _ = self.get_train_val_scenes()
            if split=='val':
                _, self.scenes = self.get_train_val_scenes()

        # budujemy jednowymiarową LUT 256^3 -> class_id
        lut = np.full((256**3,), fill_value=255, dtype=np.int64)
        for _, cid, (r, g, b) in A2D2Dataset.classes:
            lut[r * 256 * 256 + g * 256 + b] = cid
        self._lut = lut  # atrybut lookup‐table

        # lista próbek
        for scene in self.scenes:
            cam_img_dir = os.path.join(root_semantic, scene, 'camera', camera)
            for img_path in glob.glob(os.path.join(cam_img_dir, '*.png')):
                fname = os.path.basename(img_path)
                parts = fname.split('_')
                timestamp = parts[0]
                idx = parts[-1].split('.')[0]
                label_path = os.path.join(
                    root_semantic, scene, 'label', camera,
                    f"{timestamp}_label_{self.camera_id}_{idx}.png"
                )
                self.images.append(img_path)
                self.targets.append(label_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):

        # Wczytywanie PIL‐owego obrazu i PIL‐owej maski (RGB)
        img_pil = Image.open(self.images[idx]).convert('RGB')
        trg_pil = Image.open(self.targets[idx])

        if self.transforms is not None:
            image, target = self.transforms(img_pil, trg_pil)

        # Przetwarzanie PIL‐owej maski lbl_geo na NumPy, zmiana kolorów
        lbl_np = np.array(target, dtype=np.uint8)
        r = lbl_np[..., 0].astype(np.int32)
        g = lbl_np[..., 1].astype(np.int32)
        b = lbl_np[..., 2].astype(np.int32)
        idxs = (r * 256 * 256) + (g * 256) + b
        class_map = torch.from_numpy(self._lut[idxs]).long()

        return image, class_map

    def get_train_val_scenes(
            self, 
            split_idx = 17,
            seed = 42
        ):
        random.seed(seed)
        all_scenes_copy = self.scenes.copy()
        random.shuffle(all_scenes_copy)
        train_scenes = all_scenes_copy[:split_idx]
        val_scenes   = all_scenes_copy[split_idx:]
        return train_scenes, val_scenes
