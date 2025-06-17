import torch

import matplotlib.pyplot as plt
import numpy as np

def extract_palette(dataset):
    '''Generuje paletę kolorów na podstawie datasetu.'''
    return {
        cls.id: cls.color
        for cls in dataset.classes
        if getattr(cls, "train_id", 0) != 255
    }

def apply_mask_palette(mask, palette):
    '''Ustawia paletę kolorów wskazanej masce.'''
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
    for cid, rgb in palette.items():
        mask_rgb[mask_np == cid] = rgb
    return mask_rgb

def display_image_and_mask(image, mask, mask_palette):
    '''Wyświetla obraz oraz jego maskę.'''
    if isinstance(image, torch.Tensor):
        image = image.cpu()
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    img_np = np.clip(image.permute(1,2,0).numpy(), 0, 1)
    # img_np = np.clip(img_np, 0, 1)

    mask_rgb = apply_mask_palette(mask=mask, palette=mask_palette)

    plt.subplot(1, 2, 1); plt.imshow(img_np);   plt.axis('off'); plt.title("Image")
    plt.subplot(1, 2, 2); plt.imshow(mask_rgb); plt.axis('off'); plt.title("Mask")
    plt.tight_layout(); plt.show()