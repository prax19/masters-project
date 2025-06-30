import torch

import matplotlib.pyplot as plt
import numpy as np

from .encoding import ClassMapper

def display_image_and_mask(
        mapper: ClassMapper,
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
        mask_rgb = mapper.apply_mask_mapping(mask=mask_true)
        plt.subplot(1, n_images, k_image); plt.imshow(mask_rgb); plt.axis('off'); plt.title("Maska rzeczywista")
    
    if mask_pred is not None:
        k_image += 1
        mask2_rgb = mapper.apply_mask_mapping(mask=mask_pred)
        plt.subplot(1, n_images, k_image); plt.imshow(mask2_rgb); plt.axis('off'); plt.title("Maska przewidziana")
    
    plt.tight_layout(); plt.show()