import os
import warnings

import torch
from torch import autocast
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from mask_generation import generate_segmentation_mask

import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image

def replace_bn_with_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups, child.num_features)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)

def train_model(
    model,
    ckpt_name,
    train_dataset,
    test_dataset,
    device,
    batch_size = 4,
    num_epochs = 30,
    lr = 1e-4,
    weight_decay = 1e-5,
    amp = True
):
    class_names = [cls.name for cls in train_dataset.classes]
    num_classes = len(train_dataset.classes)
    
    # przygotowywanie plików
    model_root_dir=".\\model\\"
    ckpt_path = os.path.join(model_root_dir, ckpt_name + '.ckpt')
    meta_path = os.path.join(model_root_dir, ckpt_name + '.csv')

    # przygotowanie historii / odczyt z pliku
    history = pd.DataFrame(columns = ['epoch', 'train_loss', 'val_loss', 'mIoU'] + class_names)
    history.index.name = "id"
    if os.path.exists(meta_path):
        history = pd.read_csv(meta_path, index_col='id')
    
    #  -- wczytywanie zbioru --

    # dataloadery
    # parametry dobrane dla systemu z 12gb vram oraz 64gb ram
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=12, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )

    # -- 

    # if(batch_size == 1):
    #     replace_bn_with_gn(model.classifier)
    #     replace_bn_with_gn(model.aux_classifier)

    # przygotowanie modelu oraz warstw wejściowych / wyjściowych
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    model = model.to(device=device)
    model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.amp.GradScaler(device=device, enabled=amp)
    criterion = nn.CrossEntropyLoss()

    # wczytanie checkpointu
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint '{ckpt_name}'")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Pętla treningowa
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader)
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks  = masks.squeeze(1).long().to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(images)
                main_out = outputs['out']; aux_out = outputs.get('aux', None)
                loss_main = criterion(main_out, masks)
                loss = loss_main
                if aux_out is not None:
                    loss_aux = criterion(aux_out, masks)
                    loss += 0.4 * loss_aux

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{running_loss / (pbar.n + 1) / images.size(0):.4f}",  # średnia od początku epoki
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")

        # walidacja
        model.eval()
        running_val_loss = 0.0

        conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

        with torch.no_grad():
            for images, masks in tqdm(test_loader):
                images = images.to(device, non_blocking=True)
                masks  = masks.squeeze(1).long().to(device,  non_blocking=True)

                with autocast(device_type='cuda'):
                    outputs = model(images)
                    main_out = outputs['out']; aux_out = outputs.get('aux', None)
                    loss_main = criterion(main_out, masks)
                    val_loss = loss_main
                    if aux_out is not None:
                        loss_aux = criterion(aux_out, masks)
                        val_loss += 0.4 * loss_aux

                running_val_loss += val_loss.item() * images.size(0)

                preds = main_out.argmax(1)

                pred_flat = preds.view(-1)
                mask_flat = masks.view(-1)

                # Filtracja nieprawidłowych wartości
                valid = (mask_flat >= 0) & (mask_flat < num_classes)
                pred_flat = pred_flat[valid]
                mask_flat = mask_flat[valid]

                # Obliczanie macierzy pomyłek
                indices = num_classes * mask_flat + pred_flat
                conf_matrix += torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)

        # Obliczanie IoU
        val_loss_epoch = running_val_loss / len(test_loader.dataset)
        inter = torch.diag(conf_matrix).float()
        union = conf_matrix.sum(1) + conf_matrix.sum(0) - torch.diag(conf_matrix)
        ious = inter / (union + 1e-6)
        ious[union == 0] = float('nan')
        miou = torch.nanmean(ious).item()

        # Historia pomiarów
        ious_cpu = ious.detach().cpu().numpy() 
        row = {
            'epoch': float(epoch),
            'train_loss': float(epoch_loss),
            'val_loss':   float(val_loss_epoch),
            'mIoU':       float(miou)
        }

        for idx, cls in enumerate(train_dataset.classes):
            val = ious_cpu[idx]
            row[cls.name] = float(val) if not (val != val) else float('nan')
        
        history.loc[len(history)] = row
        history.to_csv(meta_path)

        print(f"[Epoch {epoch+1}/{num_epochs}] Val   Loss: {val_loss_epoch:.4f} | mIoU: {miou:.4f}")
        for cls in range(num_classes):
            classs_name = train_dataset.classes[cls].name if cls < num_classes else f"Class {cls}"
            if torch.isnan(ious[cls]):
                print(f"  {classs_name:<30s} IoU: n/a")
            else:
                print(f"  {classs_name:<30s} IoU: {ious[cls]:.4f}")

        # Zapisanie / nadpisanie checkpointu
        if os.path.exists(ckpt_path):
            new_path = os.path.join(model_root_dir, ckpt_name + "_ep_" + str(epoch - 1) + '.ckpt')
            os.rename(ckpt_path, new_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        },  ckpt_path)

        # wyświetlanie maski
        mask = generate_segmentation_mask(
            model=model,
            image=Image.open(".\\test\\test2.png").convert("RGB"),
            classes=train_dataset.classes,
            device=device
        )
        plt.figure()  
        plt.imshow(mask)
        plt.axis('off')
        plt.show()

        scheduler.step()
