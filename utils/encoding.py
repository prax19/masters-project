import albumentations as  A
from albumentations.pytorch import ToTensorV2

import torch
import numpy as np

class ClassMappeer:
    """
    Klasa odpowiadająca za mapowanie i kodowanie indeksów pikseli w zależności od zestawu klas zawartych w zbiorze danych.
    Zawiera ona wrappery biblioteki `Albumentations`, które przekazane w miejsce parametrów `transforms` Datasetów posłużą do wykonania transformacji, przeprowadzenia kodowania na podstawie zbioru klas oraz konwersji do tensora.
    Dodatkowo klasa ta dostarcza pole `mapping`, które służy jako paleta kolorów i nazw klas zbioru danych.
    """
    def __init__(
            self, 
            dataset_class, 
            train_transforms,
            infer_transforms,
            reduced_subset: bool = False
        ):
        self.train_transforms = train_transforms
        self.infer_transforms = infer_transforms
        self.reduced_subset = reduced_subset

        self.mapping = {}
        for cls in dataset_class.classes:
            if reduced_subset:
                self.mapping.setdefault(cls.train_id, (cls.id, cls.name, cls.color))
            else:
                self.mapping.setdefault(cls.id, (cls.train_id, cls.name, cls.color))

        self.id2tid = {
            cls.id: (cls.train_id if reduced_subset else cls.id)
            for cls in dataset_class.classes
        }

    def encode_indexes(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        if not self.reduced_subset:
            return mask
        out = np.full_like(mask, 255)
        for orig_id, train_id in self.id2tid.items():
            out[mask == orig_id] = train_id
        return out
    
    def wrap_train(self, img, mask):
        trans = A.Compose([
            self.train_transforms,
            A.Lambda(mask=self.encode_indexes),
            ToTensorV2()
        ])
        aug = trans(image=np.asarray(img), mask=np.asarray(mask))
        return aug["image"].float() / 255., aug["mask"].long()
    
    def wrap_infer(self, img, mask):
        trans = A.Compose([
            self.infer_transforms,
            A.Lambda(mask=self.encode_indexes),
            ToTensorV2()
        ])
        aug = trans(image=np.asarray(img), mask=np.asarray(mask))
        return aug["image"].float() / 255., aug["mask"].long()
    
    def preprocess_image(self, img):
        trans = A.Compose([
            self.infer_transforms,
            ToTensorV2()
        ])
        aug = trans(image=np.asarray(img))
        return aug["image"].float() / 255.
    
    def apply_mask_mapping(self, mask):
        '''Konfiguruje mapowanie kolorów wskazanej masce.'''
        if isinstance(mask, torch.Tensor):
            mask_arr = mask.cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask_arr = mask
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
            mask_np = mask_arr[0]
        elif mask_arr.ndim == 2:
            mask_np = mask_arr
        else:
            raise ValueError(f"Unexpected mask shape {mask_arr.shape}")
        
        mask_rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
        for cid, (_, _, rgb) in self.mapping.items():
            mask_rgb[mask_np == cid] = rgb
        return mask_rgb
    
    def state_dict(self) -> dict:
        """Zwraca JSON-owalny słownik opisujący stan."""
        return {
            "reduced_subset": self.reduced_subset,
            "mapping":        self.mapping,
            "id2tid":         self.id2tid,
            "train_tf_cfg":   A.to_dict(self.train_transforms),
            "infer_tf_cfg":   A.to_dict(self.infer_transforms),
        }
    
    def load_state_dict(self, state: dict):
        """Odtwarzanie stanu"""
        self.reduced_subset = state["reduced_subset"]
        self.mapping        = state["mapping"]
        self.id2tid         = state["id2tid"]
        self.train_transforms = A.from_dict(state["train_tf_cfg"])
        self.infer_transforms = A.from_dict(state["infer_tf_cfg"])