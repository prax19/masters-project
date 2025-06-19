import torch

import matplotlib.pyplot as plt
import numpy as np

from typing_extensions import deprecated

@deprecated("Use `transforms.IndexEncoder.mapping` instead.")
def extract_palette(dataset):
    '''Generuje paletę kolorów na podstawie datasetu.'''
    palette = {}
    for cls in dataset.classes:
        palette.setdefault(cls.train_id, cls.color)
    return palette

@deprecated("Use `apply_mask_mapping()` instead.")
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

def apply_mask_mapping(mask, mapping):
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
    for cid, (_, _, rgb) in mapping.items():
        mask_rgb[mask_np == cid] = rgb
    return mask_rgb

@deprecated("Use new `display_image_and_mask()` instead.")
def display_image_and_mask(
        mask_palette,
        image, 
        mask_true = None,
        mask_pred = None,
    ):
    '''Wyświetla obraz oraz jego maski.'''
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1,2,0).numpy()
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    n_images = 1
    if mask_true is not None:
        n_images += 1
    if mask_pred is not None:
        n_images += 1

    k_image = 1

    # Konstrukcja wykresów
    plt.figure(figsize=(12, 6))

    img_np = np.clip(image, 0, 1)
    plt.subplot(1, n_images, k_image); plt.imshow(img_np);   plt.axis('off'); plt.title("Obraz")

    if mask_true is not None:
        k_image += 1
        mask_rgb = apply_mask_palette(mask=mask_true, palette=mask_palette)
        plt.subplot(1, n_images, k_image); plt.imshow(mask_rgb); plt.axis('off'); plt.title("Maska rzeczywista")
    
    if mask_pred is not None:
        k_image += 1
        mask2_rgb = apply_mask_palette(mask=mask_pred, palette=mask_palette)
        plt.subplot(1, n_images, k_image); plt.imshow(mask2_rgb); plt.axis('off'); plt.title("Maska przewidziana")
    
    plt.tight_layout(); plt.show()

def display_image_and_mask(
        mapping,
        image, 
        mask_true = None,
        mask_pred = None,
    ):
    '''Wyświetla obraz oraz jego maski.'''
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1,2,0).numpy()
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    n_images = 1
    if mask_true is not None:
        n_images += 1
    if mask_pred is not None:
        n_images += 1

    k_image = 1

    # Konstrukcja wykresów
    plt.figure(figsize=(12, 6))

    img_np = np.clip(image, 0, 1)
    plt.subplot(1, n_images, k_image); plt.imshow(img_np);   plt.axis('off'); plt.title("Obraz")

    if mask_true is not None:
        k_image += 1
        mask_rgb = apply_mask_mapping(mask=mask_true, mapping=mapping)
        plt.subplot(1, n_images, k_image); plt.imshow(mask_rgb); plt.axis('off'); plt.title("Maska rzeczywista")
    
    if mask_pred is not None:
        k_image += 1
        mask2_rgb = apply_mask_mapping(mask=mask_pred, mapping=mapping)
        plt.subplot(1, n_images, k_image); plt.imshow(mask2_rgb); plt.axis('off'); plt.title("Maska przewidziana")
    
    plt.tight_layout(); plt.show()