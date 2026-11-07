"""g2pS runtime: self-contained ONNX inference for Mandarin polyphone disambiguation.

Deps: onnxruntime, numpy. No torch, no transformers.
"""
import os
import pickle

import numpy as np
import onnxruntime as ort

MAX_LEN = 128


class G2pS:
    def __init__(self, model_dir, use_int8=True, providers=None):
        meta = pickle.load(open(os.path.join(model_dir, 'meta.pkl'), 'rb'))
        self.vocab = meta['vocab']                    # char -> id (id 1 = UNK)
        self.labels = meta['labels']                  # pinyin strings, e.g. 'hang2'
        self.label_id = {l: i for i, l in enumerate(self.labels)}
        self.poly_chars = meta['poly_chars']          # chars the model disambiguates
        self.char_cands = meta['char_cands']          # char -> list of candidate pinyins
        self.poly_id = {c: i for i, c in enumerate(self.poly_chars)}
        self.n_labels = len(self.labels)

        onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
        if not onnx_files:
            raise FileNotFoundError(
                f'no .onnx found in {model_dir} — run `bash scripts/download_weights.sh` '
                'or fetch weights from HF/ModelScope (see README)')
        if use_int8:
            cand = [f for f in onnx_files if '.int8.' in f] or onnx_files
        else:
            cand = [f for f in onnx_files if '.int8.' not in f] or onnx_files
        path = os.path.join(model_dir, sorted(cand)[0])

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=providers or ['CPUExecutionProvider'])
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def _prepare(self, text):
        text = text.lower()  # training data was lowercased
        positions = [i for i, c in enumerate(text) if c in self.poly_id]
        if not positions:
            return None
        if len(text) > MAX_LEN:
            first = positions[0]
            start = max(0, min(first - MAX_LEN // 2, len(text) - MAX_LEN))
            text = text[start:start + MAX_LEN]
            positions = [p - start for p in positions if start <= p < start + MAX_LEN]
            if not positions:
                return None
        char_ids = np.array([[self.vocab.get(c, 1) for c in text]], dtype=np.int64)
        P = len(positions)
        poly_ids = np.array([[self.poly_id[text[p]] for p in positions]], dtype=np.int64)
        pos_arr = np.array([positions], dtype=np.int64)
        cand = np.zeros((1, P, self.n_labels), dtype=bool)
        for k, p in enumerate(positions):
            cand[0, k, list(self.char_cands[text[p]])] = True  # values are label ids
        return text, positions, char_ids, poly_ids, pos_arr, cand

    def predict(self, text):
        """Returns {position: pinyin} over the *preprocessed* text (lowercased, maybe truncated).

        Use convert() for a char-aligned output."""
        prep = self._prepare(text)
        if prep is None:
            return {}, text
        text, positions, char_ids, poly_ids, pos_arr, cand = prep
        feeds = {'char_ids': char_ids, 'poly_ids': poly_ids, 'positions': pos_arr, 'cand_mask': cand}
        logp = self.sess.run(None, feeds)[0]
        pred = logp.argmax(-1)[0]
        return {p: self.labels[int(pred[k])] for k, p in enumerate(positions)}, text

    def convert(self, text):
        """Returns (text_used, [(char, pinyin_or_None), ...]).
        Monophonic chars get None — pair with pypinyin if you need full-sentence pinyin."""
        res, t = self.predict(text)
        return t, [(c, res.get(i)) for i, c in enumerate(t)]


if __name__ == '__main__':
    import sys
    g = G2pS(sys.argv[1] if len(sys.argv) > 1 else 'models/nano')
    for s in ['他在银行工作，行踪不定。', '恢复汉阳区人民政府', '能把小孩哄睡是个技术活。']:
        _, pairs = g.convert(s)
        print(s, '->', [(c, p) for c, p in pairs if p])
