import albumentations as  A
from albumentations.pytorch import ToTensorV2

import cv2

import numpy as np

transforms_test = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.8, 1.2), 
        rotate=(-8, 8), 
        translate_percent=(0.1, 0.1), 
        interpolation=cv2.INTER_NEAREST, 
        mask_interpolation=cv2.INTER_NEAREST, 
        border_mode=cv2.BORDER_REFLECT, 
        fill=0, 
        fill_mask=255, 
        p=1.0
    ),
    A.PadIfNeeded(min_height=768, min_width=768, border_mode=cv2.BORDER_REFLECT),
    A.RandomCrop(height=768, width=768)
])

transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.8, 1.2), 
        rotate=(-8, 8), 
        translate_percent=(0.1, 0.1), 
        interpolation=cv2.INTER_NEAREST, 
        mask_interpolation=cv2.INTER_NEAREST, 
        border_mode=cv2.BORDER_REFLECT, 
        fill=0, 
        fill_mask=255, 
        p=1.0
    ),
    A.PadIfNeeded(min_height=768, min_width=768, border_mode=cv2.BORDER_REFLECT),
    A.RandomCrop(height=768, width=768),
    A.GaussNoise(p=0.2),
    A.Perspective(p=0.5),
    A.OneOf(
        [
            A.CLAHE(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            A.RandomGamma(p=1.0),
        ],
        p=0.9,
    ),
    A.OneOf(
        [
            A.Sharpen(p=1.0),
            A.Blur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ],
        p=0.9,
    ),
    A.OneOf(
        [
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
        ],
        p=0.9,
    )
])

class IndexEncoder:
    """
    Klasa odpowiadająca za mapowanie `train_id` oraz zapewnienie zgodności augmentacji
    z biblioteką Albumentations.
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