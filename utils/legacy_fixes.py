import glob, os
import torch
from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus

def find_checkpoints(logs_root):
    checkpoint_files = {}
    logs = glob.glob(os.path.join(logs_root, '*'))
    for log in logs:
        versions = glob.glob(os.path.join(log, 'version_*'))
        for version_pth in versions:
            version = version_pth.split('_')[1]
            checkpoints = glob.glob(os.path.join(version_pth, 'checkpoints', '*.ckpt'))
            for checkpoint in checkpoints:
                checkpoint_files.update({checkpoint: [version]})
    return checkpoint_files

def fix_checkpoint(old_path: str, new_path: str,
                   weight_key='model.segmentation_head.0.weight',
                   bias_key='model.segmentation_head.0.bias'):
    torch.serialization.add_safe_globals([DeepLabV3Plus])
    ckpt = torch.load(old_path, map_location='cpu', weights_only=False)

    state = ckpt.get('state_dict', ckpt)
    w = state[weight_key]; b = state[bias_key]
    C_old = w.shape[0]
    if b.shape[0] != C_old:
        raise ValueError(f"Mismatched channels: {C_old} vs {b.shape[0]}")
    
    state[weight_key] = w[:C_old-1].clone()
    state[bias_key]   = b[:C_old-1].clone()

    fixed = ckpt.copy()
    fixed['state_dict'] = state
    torch.save(fixed, new_path)
    print(f"Checkpoint poprawiony: {new_path}")

def fix_checkpoint_bulk(root):
    ckpts = find_checkpoints(root)

    for ckpt in ckpts:
        name, ext = os.path.splitext(ckpt)
        fix_checkpoint(ckpt, f'{name}_fixed{ext}')