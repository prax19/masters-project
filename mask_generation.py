import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing_extensions import deprecated

import torchvision.transforms.v2 as v2
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

import numpy as np
from PIL import Image

@deprecated("An old experimental method.")
def replace_bn_with_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups, child.num_features)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)

@deprecated("An old experimental method.")
def generate_segmentation_mask(
        model,
        image: Image,
        classes,
        device,
        transform = None
    ):
    model.eval()
    orig_width, orig_height = image.size

    transform or v2.Compose([
        v2.PILToTensor(),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']
        prediction = output.argmax(1).squeeze(0).cpu().numpy()

    # Załadowywanie palety (mapa class_id -> RGB)
    palette = {
        cls.id: cls.color
        for cls in classes
        if getattr(cls, "train_id", 0) != 255
    }

    # Kolorowy obraz predykcji
    color_pred = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for cls_id, rgb in palette.items():
        color_pred[prediction == cls_id] = rgb

    # Wyjściowa interpolacja
    cp_t = torch.from_numpy(color_pred).permute(2, 0, 1).unsqueeze(0).float().to(device)
    cp_up = F.interpolate(cp_t,
                        size=(orig_height, orig_width),
                        mode='nearest')
    
    # z powrotem na numpy [H, W, 3]
    color_pred_interpolated = cp_up.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(color_pred_interpolated)

@deprecated("An old experimental method.")
def generate_segmentation_mask_file(
        checkpoint_path, 
        input_img_path, 
        classes,
        device,
        model = None,
        model_arch = deeplabv3_resnet50,
        weights = DeepLabV3_ResNet50_Weights.DEFAULT,

    ):
    num_classes = len(classes)

    if model == None:
        model = model_arch(weights=weights)
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

        # dla batch_size = 1
        # if(1 == 1):
        #     replace_bn_with_gn(model.classifier)
        #     replace_bn_with_gn(model.aux_classifier)

    model = model.to(device=device)
    model = torch.compile(model)

    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt['epoch']

    image = Image.open(input_img_path).convert("RGB")
    mask = generate_segmentation_mask(
        model=model,
        image=image,
        classes=classes,
        device=device
    )

    # zapis do pliku
    base, ext = os.path.splitext(input_img_path)
    new_path = base + "_pred8_ep" + str(epoch) + ext
    mask.save(new_path)

    return mask

@deprecated("An old experimental method.")
def generate_segmentation_mask_tiled(
    checkpoint_path: str,
    input_img_path: str,
    output_path: str,
    list_json_path: str = ".\\data\\camera_lidar_semantic\\class_list.json",
    tile_size: tuple[int, int] = (768, 768),
    overlap: int = 128
) -> Image.Image:
    """
    Wczytuje model z checkpointu, dzieli dowolnej wielkości obraz na nakładające się kafle,
    uruchamia na każdym kaflu segmentację, skleja wynikową maskę w jedną pełno-rozmiarową
    kolorową mapę klas i zapisuje ją pod output_path. Zwraca wygenerowany PIL.Image.
    
    Args:
      checkpoint_path: ścieżka do .pth z kluczem 'model_state_dict'.
      input_img_path:  ścieżka do dowolnego obrazu RGB.
      output_path:     dokąd zapisać kolorową maskę.
      list_json_path:  ścieżka do pliku class_list.json (format: {"#HEX": "class_name", ...}).
      tile_size:       (wysokość, szerokość) pojedynczego kafla.
      overlap:         liczba pikseli nakładki w pionie i poziomie.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Używamy architektury DeepLabV3-ResNet50
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)

    # wczytywanie checkpointu
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Odczytaj liczbę klas z kształtu wagi ostatniej warstwy (classifier.4.weight)
    w_cls = state.get("classifier.4.weight", None)
    if w_cls is not None:
        num_classes = w_cls.shape[0]
    else:
        # Jeśli nie ma klasycznego klucza, weź liczbę z pretrenowanych meta-danych
        num_classes = len(weights.meta["categories"])

    # Nadpisz ostatnie warstwy, by pasowały do num_classes
    model.classifier[-1]    = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    model.load_state_dict(state)
    model.to(device).eval()

    #  odczytanie palety koloru
    with open(list_json_path, "r") as f:
        hex2name = json.load(f)


    id_to_rgb = {}
    for class_id, (hex_col, cls_name) in enumerate(hex2name.items()):
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        id_to_rgb[class_id] = (r, g, b)

    # odczytanie obrazu wejściowego
    img_pil = Image.open(input_img_path).convert("RGB")
    orig_w, orig_h = img_pil.size
    tile_h, tile_w = tile_size

    # transformacje kafla
    tile_transform = v2.Compose([
        v2.PILToTensor(),
        v2.ConvertImageDtype(torch.float32),
    ])

    # pusta tablica wyniku
    result = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    # funkcja do obliczeń kaflowania
    def compute_tile_coords(img_w: int, img_h: int, tw: int, th: int, ov: int):
        xs, ys = [], []
        step_x = tw - ov
        step_y = th - ov

        x = 0
        while x + tw < img_w:
            xs.append(x)
            x += step_x
        xs.append(max(0, img_w - tw))

        y = 0
        while y + th < img_h:
            ys.append(y)
            y += step_y
        ys.append(max(0, img_h - th))

        return xs, ys

    xs, ys = compute_tile_coords(orig_w, orig_h, tile_w, tile_h, overlap)

    # iteracja po kaflach
    logits_accum = torch.zeros(num_classes, orig_h, orig_w, device="cpu")
    counts       = torch.zeros(orig_h, orig_w,               device="cpu")

    with torch.no_grad():
        for y in ys:
            for x in xs:
                box  = (x, y, x + tile_w, y + tile_h)
                tile = img_pil.crop(box)
                w_c, h_c = tile.size         # <--- brzegowy kafel?

                # PAD zamiast resize
                if (w_c != tile_w) or (h_c != tile_h):
                    pad_w = tile_w - w_c
                    pad_h = tile_h - h_c
                    tile  = v2.functional.pad(tile, (0, 0, pad_w, pad_h),
                                            padding_mode="edge")

                inp   = tile_transform(tile).unsqueeze(0).to(device)
                out   = model(inp)["out"].cpu().squeeze(0)      # [C,H,W]

                # MEAN-blending logitów
                logits_accum[:, y:y+tile_h, x:x+tile_w] += out
                counts      [y:y+tile_h, x:x+tile_w]    += 1

    # argmax na zblendowanych logitach
    logits_accum /= counts.unsqueeze(0)
    pred_full     = logits_accum.argmax(0).numpy()    # [H,W]

    # kolorowanie
    result = np.zeros((orig_h, orig_w, 3), np.uint8)
    for cls_id, rgb in id_to_rgb.items():
        result[pred_full == cls_id] = rgb

    # konwersja na PIL i zapis
    result_img = Image.fromarray(result)
    result_img.save(output_path)
    return result_img
