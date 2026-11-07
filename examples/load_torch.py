"""Load PyTorch weights with the packaged architecture definition."""
import pickle
import torch
from g2ps.model import StudentG2P

ARCH = dict(
    small=dict(d=384, n_layers=6, n_heads=6, ffn_hidden=1024, emb_dim=None, bias_rank=0),
    tiny=dict(d=256, n_layers=5, n_heads=4, ffn_hidden=640, emb_dim=128, bias_rank=64),
    nano=dict(d=96, n_layers=6, n_heads=4, ffn_hidden=256, emb_dim=48, bias_rank=16, n_passes=2),
)

def load(tier, root='.'):
    meta = pickle.load(open(f'{root}/models/{tier}/meta.pkl', 'rb'))
    kw = dict(ARCH[tier])
    np_ = kw.pop('n_passes', 1)
    m = StudentG2P(len(meta['vocab']), len(meta['labels']), len(meta['poly_chars']),
                   max_len=128, use_conv_stem=True, n_passes=np_, **kw)
    m.load_state_dict(torch.load(f'{root}/torch_weights/{tier}/{tier}.pt', map_location='cpu'))
    m.eval()
    return m, meta

if __name__ == '__main__':
    for tier in ['small', 'tiny', 'nano']:
        m, meta = load(tier)
        print(tier, 'loaded:', sum(p.numel() for p in m.parameters()) / 1e6, 'M params')
