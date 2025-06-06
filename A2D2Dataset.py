import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as v2

class A2D2SegmentationDataset(Dataset):
    def __init__(
            self,
            root_dir,
            scenes=None,
            camera='cam_front_center',
            transforms_geo=None,
            transforms_img=None,
            with_pointcloud=False,
            with_meta=False
    ):
        self.root_dir = root_dir
        self.camera = camera
        self.camera_id = camera.replace('cam_', '').replace('_', '')
        self.transforms_geo = transforms_geo or v2.Compose([ 
            v2.RandomResizedCrop((768, 768), scale=(0.5, 2.0)),
            v2.RandomHorizontalFlip(p=0.5)
        ])
        self.transforms_img = transforms_img or v2.Compose([ 
            v2.ColorJitter(0.3, 0.3, 0.3, 0.1),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.with_pointcloud = with_pointcloud
        self.with_meta = with_meta

        # wczytywanie mapping HEX->nazwa z JSON i zbuduj PALETTE: (R,G,B)->class_id
        palette_path = os.path.join(root_dir, 'class_list.json')
        with open(palette_path, 'r') as f:
            hex2name = json.load(f)

        # zachowujemy insertion order, numerujemy klasy kolejno
        self.PALETTE = {}
        self.class_names = []
        for class_id, (hex_col, name) in enumerate(hex2name.items()):
            rgb = tuple(int(hex_col.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            self.PALETTE[rgb] = class_id
            self.class_names.append(name)

        # budujemy jednowymiarową LUT 256^3 -> class_id
        lut = np.zeros((256**3,), dtype=np.int64)
        for (r, g, b), cid in self.PALETTE.items():
            lut[r * 256 * 256 + g * 256 + b] = cid
        self._lut = lut  # atrybut lookup‐table

        # lista próbek
        scene_dirs = scenes or sorted(os.listdir(root_dir))
        self.samples = []
        for scene in scene_dirs:
            cam_img_dir = os.path.join(root_dir, scene, 'camera', camera)
            for img_path in glob.glob(os.path.join(cam_img_dir, '*.png')):
                fname = os.path.basename(img_path)
                parts = fname.split('_')
                timestamp = parts[0]
                idx = parts[-1].split('.')[0]
                label_path = os.path.join(
                    root_dir, scene, 'label', camera,
                    f"{timestamp}_label_{self.camera_id}_{idx}.png"
                )
                sample = {'image': img_path, 'label': label_path}
                if self.with_pointcloud:
                    pc_path = os.path.join(
                        root_dir, scene, 'lidar', camera,
                        f"{timestamp}_lidar_{self.camera_id}_{idx}.npz"
                    )
                    sample['pointcloud'] = pc_path
                if self.with_meta:
                    json_path = os.path.join(
                        root_dir, scene, 'camera', camera,
                        f"{timestamp}_camera_{self.camera_id}_{idx}.json"
                    )
                    sample['meta'] = json_path
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Wczytywanie PIL‐owego obrazu i PIL‐owej maski (RGB)
        img_pil = Image.open(sample['image']).convert('RGB')
        lbl_pil = Image.open(sample['label']).convert('RGB')

        # Wspólne transformacje geometryczne (obraz + maska)
        img_geo, lbl_geo = self.transforms_geo(img_pil, lbl_pil)

        # Transformacje na obrazie
        img_tensor = self.transforms_img(img_geo)

        # Przetwarzanie PIL‐owej maski lbl_geo na NumPy, zmiana kolorów
        lbl_np = np.array(lbl_geo, dtype=np.uint8)
        r = lbl_np[..., 0].astype(np.int32)
        g = lbl_np[..., 1].astype(np.int32)
        b = lbl_np[..., 2].astype(np.int32)
        idxs = (r * 256 * 256) + (g * 256) + b
        class_map = self._lut[idxs]
        lbl_tensor = torch.from_numpy(class_map).long()

        return {'image': img_tensor, 'label': lbl_tensor}
