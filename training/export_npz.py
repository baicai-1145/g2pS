"""Export trained StudentG2P checkpoint to numpy-only inference weights.

Usage: python training/export_npz.py checkpoints/nano_best.pt datasets/student_v5/meta.pkl models/nano/nano.npz
"""
import pickle
import sys

import numpy as np
import torch

src_pt, meta_pkl, dst_npz = sys.argv[1:4]
sd = torch.load(src_pt, map_location='cpu')
sd = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in sd.items()}
sd = {k: v for k, v in sd.items() if not k.startswith(('syl_head', 'tone_head'))}
np.savez(dst_npz, **{k: v.numpy().astype(np.float16) for k, v in sd.items()})
print(f'{dst_npz}: {sum(v.numel() for v in sd.values())/1e6:.1f}M params, fp16 storage')
# sanity: meta travels with the model
import os
meta_dst = os.path.join(os.path.dirname(dst_npz), 'meta.pkl')
if not os.path.exists(meta_dst):
    import shutil
    shutil.copy(meta_pkl, meta_dst)
    print('copied meta.pkl alongside')
