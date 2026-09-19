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

qcheck = df['s_query'].iloc[0]
toks0 = [str(x) for x in str(qcheck).split()]
print('пример s_query:', repr(str(df['s_query'].iloc[0]))[:80],
      '| в словаре:', sum(1 for t in toks0 if t in ti), '/', len(toks0))

new_feats = []
for qi in range(len(feats)):
    ids, X, t = feats[qi]
    qterms = [ti.get(x) for x in str(df['s_query'].iloc[qi]).split()]
    qterms = sorted(set(int(c) for c in qterms if c is not None))
    if not qterms:
        new_feats.append(np.zeros(len(ids), dtype=np.float32))
        continue
    wsum = float(np.sum(idf[qterms]))
    cur = np.zeros(len(ids), dtype=np.float32)
    for c in qterms:
        s, e = full.indptr[c], full.indptr[c + 1]
        hit = np.isin(ids, full.indices[s:e])
        cur[hit] += idf[c] if wsum > 0 else 1.0
    cur = cur / wsum
    new_feats.append(cur.astype(np.float32))


def rec(we):
    tot = 0.0
    n = 0
    for qi in range(len(feats)):
        ids, X, t = feats[qi]
        if len(ids) == 0:
            continue
        n += 1
        k = min(50, len(ids))
        sc = X[..., :10].dot(we[:10]) + we[10] * new_feats[qi]
        top = np.argpartition(-sc, k - 1)[:k]
        tot += t[top].sum() / max(1, int(t.sum()))
    return tot / n


base = np.array([2.2, 0.02, 0, 0.22, 0, 0.57, 0.25, 3.0, 0.6, 0.4], dtype=np.float32)
print('base fused:', round(rec(np.r_[base, 0.0]), 4))
print('with cov x{0.5,1,2,4}:', [round(rec(np.r_[base, g]), 4) for g in (0.5, 1.0, 2.0, 4.0)])