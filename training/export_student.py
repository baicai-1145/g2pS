"""Export a student checkpoint to ONNX (fp32 + int8), verify accuracy and CPU latency."""
import argparse
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from student.modules import StudentG2P
from student.train import collate, RecordDataset
from student.data import MAX_LEN


def build_model(args, n_vocab, n_labels, n_poly):
    return StudentG2P(n_vocab, n_labels, n_poly, d=args.d, n_layers=args.layers,
                      n_heads=args.heads, ffn_hidden=args.ffn, max_len=MAX_LEN,
                      use_conv_stem=not args.no_conv, emb_dim=(args.emb_dim or None),
                      bias_rank=args.bias_rank)


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, char_ids, poly_ids, positions, cand_mask):
        z = torch.zeros_like(positions, dtype=torch.bool)
        logp, _, _, _, _ = self.m.forward_with_ids(char_ids, poly_ids, positions, z, cand_mask.bool())
        return logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data_root', default='datasets/student_v4')
    ap.add_argument('--d', type=int, default=384)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--ffn', type=int, default=1024)
    ap.add_argument('--emb_dim', type=int, default=0)
    ap.add_argument('--bias_rank', type=int, default=0)
    ap.add_argument('--no_conv', action='store_true')
    args = ap.parse_args()

    meta = pickle.load(open(f'{args.data_root}/meta.pkl', 'rb'))
    vocab, labels, poly_chars, char_cands = meta['vocab'], meta['labels'], meta['poly_chars'], meta['char_cands']
    n_labels = len(labels)
    char_cand_mask = {i: torch.zeros(n_labels, dtype=torch.bool).index_fill_(0, torch.tensor(sorted(char_cands[ch])), True)
                      for i, ch in enumerate(poly_chars)}
    model = build_model(args, len(vocab), n_labels, len(poly_chars))
    sd = torch.load(args.ckpt, map_location='cpu')
    sd = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith(('syl_head', 'tone_head'))}
    model.load_state_dict(sd)
    model.eval()

    test = pickle.load(open(f'{args.data_root}/test.pkl', 'rb'))

    # torch fp32 accuracy
    def acc_of(m):
        m.eval()
        dl = DataLoader(RecordDataset(test), batch_size=256, collate_fn=lambda b: collate(b, n_labels, char_cand_mask))
        correct = total = 0
        with torch.no_grad():
            for b in dl:
                char_ids, poly_ids, positions, pos_mask, cand_mask, label_ids, pos_ids = b
                logp, _, _, _, _ = m.forward_with_ids(char_ids, poly_ids, positions, pos_mask, cand_mask)
                pred = logp.argmax(-1)
                correct += ((pred == label_ids) & pos_mask).sum().item()
                total += pos_mask.sum().item()
        return correct / total

    acc_fp = acc_of(model)
    print(f'torch fp32 test acc: {acc_fp:.4f}')

    # export onnx
    wrap = ExportWrapper(model)
    B, T, P = 1, 32, 8
    dummy = (torch.ones(B, T, dtype=torch.int64), torch.zeros(B, P, dtype=torch.int64),
             torch.full((B, P), 5, dtype=torch.int64),
             torch.ones(B, P, n_labels, dtype=torch.bool))
    torch.onnx.export(wrap, dummy, f'{args.out}.onnx', opset_version=17, dynamo=False,
                      input_names=['char_ids', 'poly_ids', 'positions', 'cand_mask'],
                      output_names=['logp'], dynamic_axes={
                          'char_ids': {0: 'B', 1: 'T'}, 'poly_ids': {0: 'B', 1: 'P'},
                          'positions': {0: 'B', 1: 'P'}, 'cand_mask': {0: 'B', 1: 'P'},
                          'logp': {0: 'B', 1: 'P'}})
    print('exported', f'{args.out}.onnx')

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(f'{args.out}.onnx', so, providers=['CPUExecutionProvider'])
    input_names = {i.name for i in sess.get_inputs()}
    def feeds(b):
        d = {'char_ids': b[0].numpy(), 'poly_ids': b[1].numpy(), 'positions': b[2].numpy(),
             'cand_mask': b[4].numpy()}
        return {k: v for k, v in d.items() if k in input_names}
    # verify onnx accuracy on a slice
    correct = total = 0
    dl = DataLoader(RecordDataset(test[:2000]), batch_size=64, collate_fn=lambda b: collate(b, n_labels, char_cand_mask))
    with torch.no_grad():
        for b in dl:
            char_ids, poly_ids, positions, pos_mask, cand_mask, label_ids, pos_ids = b
            out = sess.run(None, feeds(b))[0]
            pred = out.argmax(-1)
            correct += ((pred == label_ids.numpy()) & pos_mask.numpy()).sum()
            total += pos_mask.numpy().sum()
    print(f'onnx fp32 acc (slice 2000): {correct/total:.4f}')

    # latency
    b1 = next(iter(DataLoader(RecordDataset(test[:50]), batch_size=1,
                              collate_fn=lambda b: collate(b, n_labels, char_cand_mask))))
    feeds1 = feeds(b1)
    for _ in range(3):
        sess.run(None, feeds1)
    t0 = time.time()
    for _ in range(50):
        sess.run(None, feeds1)
    print(f'onnx fp32 1-thread-ish latency: {(time.time()-t0)/50*1000:.1f} ms/sent')

    # int8 dynamic quantization
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(f'{args.out}.onnx', f'{args.out}.int8.onnx', weight_type=QuantType.QInt8)
    import os
    print(f"fp32 size: {os.path.getsize(args.out+'.onnx')/1e6:.1f} MB | int8 size: {os.path.getsize(args.out+'.int8.onnx')/1e6:.1f} MB")
    sess8 = ort.InferenceSession(f'{args.out}.int8.onnx', so, providers=['CPUExecutionProvider'])
    correct = total = 0
    for b in dl:
        char_ids, poly_ids, positions, pos_mask, cand_mask, label_ids, pos_ids = b
        out = sess8.run(None, feeds(b))[0]
        pred = out.argmax(-1)
        correct += ((pred == label_ids.numpy()) & pos_mask.numpy()).sum()
        total += pos_mask.numpy().sum()
    print(f'onnx int8 acc (slice): {correct/total:.4f}')
    for _ in range(3):
        sess8.run(None, feeds1)
    t0 = time.time()
    for _ in range(50):
        sess8.run(None, feeds1)
    print(f'onnx int8 latency: {(time.time()-t0)/50*1000:.1f} ms/sent')


if __name__ == '__main__':
    main()
