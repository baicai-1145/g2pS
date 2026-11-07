import argparse, os, time
import torch
from torch.utils.data import DataLoader
from student.train import collate, RecordDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--data_root', default='datasets/student_v4')
    args = ap.parse_args()
    import pickle
    meta = pickle.load(open(f'{args.data_root}/meta.pkl', 'rb'))
    labels_n = len(meta['labels']); char_cands = meta['char_cands']; poly_chars = meta['poly_chars']
    char_cand_mask = {i: torch.zeros(labels_n, dtype=torch.bool).index_fill_(0, torch.tensor(sorted(char_cands[ch])), True)
                      for i, ch in enumerate(poly_chars)}
    test = pickle.load(open(f'{args.data_root}/test.pkl', 'rb'))

    import onnxruntime as ort
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=['CPUExecutionProvider'])
    names = {i.name for i in sess.get_inputs()}
    def feeds(b):
        d = {'char_ids': b[0].numpy(), 'poly_ids': b[1].numpy(), 'positions': b[2].numpy(), 'cand_mask': b[4].numpy()}
        return {k: v for k, v in d.items() if k in names}

    correct = total = 0
    dl = DataLoader(RecordDataset(test), batch_size=128, collate_fn=lambda b: collate(b, labels_n, char_cand_mask))
    for b in dl:
        out = sess.run(None, feeds(b))[0]
        pred = out.argmax(-1)
        correct += ((pred == b[5].numpy()) & b[3].numpy()).sum(); total += b[3].numpy().sum()
    print(f'{os.path.basename(args.onnx)}: FULL test acc = {correct/total:.4f}')

    b1 = next(iter(DataLoader(RecordDataset(test[:50]), batch_size=1, collate_fn=lambda b: collate(b, labels_n, char_cand_mask))))
    f1 = feeds(b1)
    for _ in range(5): sess.run(None, f1)
    t0 = time.time()
    for _ in range(100): sess.run(None, f1)
    print(f'  fp32 latency: {(time.time()-t0)/100*1000:.2f} ms/sent')

    if not args.onnx.endswith('.int8.onnx'):
        from onnxruntime.quantization import quantize_dynamic, QuantType
        out8 = args.onnx.replace('.onnx', '.int8.onnx')
        quantize_dynamic(args.onnx, out8, weight_type=QuantType.QInt8)
        print(f'  fp32 size: {os.path.getsize(args.onnx)/1e6:.1f} MB | int8: {os.path.getsize(out8)/1e6:.1f} MB')
        sess8 = ort.InferenceSession(out8, so, providers=['CPUExecutionProvider'])
        names8 = {i.name for i in sess8.get_inputs()}
        f8 = feeds(b1)
        correct = total = 0
        for b in dl:
            out = sess8.run(None, feeds(b))[0]
            pred = out.argmax(-1)
            correct += ((pred == b[5].numpy()) & b[3].numpy()).sum(); total += b[3].numpy().sum()
        print(f'  int8 FULL test acc = {correct/total:.4f}')
        for _ in range(5): sess8.run(None, f8)
        t0 = time.time()
        for _ in range(100): sess8.run(None, f8)
        print(f'  int8 latency: {(time.time()-t0)/100*1000:.2f} ms/sent')

if __name__ == '__main__':
    main()
