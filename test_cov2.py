import pickle, numpy as np, pandas as pd, scipy.sparse as sp, sys
sys.path.insert(0, r'C:\Users\Лена\Documents\Default Project\avito_solution')
from config import CACHE, OUT

with open(OUT + r'\val_feats.pkl', 'rb') as f:
    df, truth, feats = pickle.load(f)
with open(CACHE + r'\token_index.pkl', 'rb') as f:
    ti = pickle.load(f)
full = sp.load_npz(CACHE + r'\full_mat.npz').tocsc()
N = full.shape[0]
dfq = np.diff(full.indptr)
idf = np.log((N - dfq + 0.5) / (dfq + 0.5) + 1.0)


def cov_feats(use_infm):
    res = []
    for qi in range(len(feats)):
        ids, X, t = feats[qi]
        text = str(df['s_query'].iloc[qi])
        if use_infm:
            text += ' ' + str(df['s_infm'].iloc[qi])
        qterms = [ti.get(x) for x in text.split()]
        qterms = sorted(set(int(c) for c in qterms if c is not None))
        if not qterms:
            res.append(np.zeros(len(ids), dtype=np.float32))
            continue
        wsum = float(np.sum(idf[qterms]))
        cur = np.zeros(len(ids), dtype=np.float32)
        for c in qterms:
            s, e = full.indptr[c], full.indptr[c + 1]
            hit = np.isin(ids, full.indices[s:e])
            cur[hit] += idf[c] if wsum > 0 else 1.0
        cur = cur / wsum
        res.append(cur.astype(np.float32))
    return res


cov_q = cov_feats(False)
cov_qi = cov_feats(True)


def rec(we):
    tot = 0.0
    n = 0
    for qi in range(len(feats)):
        ids, X, t = feats[qi]
        if len(ids) == 0:
            continue
        n += 1
        k = min(50, len(ids))
        sc = X[..., :10].dot(we[:10]) + we[10] * cov_q[qi] + we[11] * cov_qi[qi]
        top = np.argpartition(-sc, k - 1)[:k]
        tot += t[top].sum() / max(1, int(t.sum()))
    return tot / n


base = np.array([2.2, 0.02, 0, 0.22, 0, 0.57, 0.25, 3.0, 0.6, 0.4], dtype=np.float32)
best = (0, None)
cand = []
for w10 in (0.0, 0.3, 0.5, 0.7):
    for w11 in (0.0, 0.2, 0.4):
        w = np.r_[base, w10, w11]
        r = rec(w)
        cand.append((r, (w10, w11)))
        if r > best[0]:
            best = (r, (w10, w11))
print('grid cov done', round(best[0], 4), best[1])
for r, c in sorted(cand, reverse=True)[:8]:
    print(round(r, 4), c)

# совместный точечный перебор вокруг базовых весов
best = (best[0], list(best[1]))
w = np.r_[base, best[1][0], best[1][1]]
for w0 in (1.8, 2.0, 2.2, 2.4, 2.6):
    for w5 in (0.4, 0.5, 0.6, 0.7):
        for w6 in (0.0, 0.2, 0.25, 0.35):
            for w7 in (2.8, 3.0, 3.2):
                for w8 in (0.5, 0.6, 0.7):
                    cand_w = w.copy()
                    cand_w[0], cand_w[5], cand_w[6], cand_w[7], cand_w[8] = w0, w5, w6, w7, w8
                    r = rec(cand_w)
                    if r > best[0]:
                        best = (r, cand_w.tolist())
print('FINAL best:', round(best[0], 4))
print('weights:', [round(x, 4) for x in best[1]])