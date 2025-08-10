import albumentations as  A
import cv2

SIZE_DIVISOR: int = 16

train_transforms = A.Compose([
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

val_transforms = A.Compose([
    A.PadIfNeeded(
        min_height=None, min_width=None, 
        pad_height_divisor=SIZE_DIVISOR, 
        pad_width_divisor=SIZE_DIVISOR, 
        border_mode=cv2.BORDER_CONSTANT,
        fill=255,
        fill_mask=255
    ),
])

test_transforms = val_transforms