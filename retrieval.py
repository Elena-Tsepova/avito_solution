# -*- coding: utf-8 -*-
"""
Этап 3: компоненты поиска (общие для валидации и инференса).
Каналы кандидатов на запрос:
  bmF/bmT   — BM25 по полному тексту / по заголовку объявления
  dense     — косинусная близость эмбеддингов запроса и объявления
  transfer  — кликнутые в train объявления похожих запросов
  exact     — клики по точному тексту запроса (если он есть в train)
  mc        — объявления из микрокатегорий кликнутых товаров похожих запросов
  title_all — объявления, где в заголовке есть ВСЕ слова запроса
  pop       — самые часто кликавшиеся в train товары

Счёта считаются по одному запросу за раз, чтобы уложиться в ~8 ГБ RAM.
"""
import sys, pickle, os, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sknorm

sys.stdout.reconfigure(encoding='utf-8')

from config import CACHE, OUT, BASE
os.makedirs(OUT, exist_ok=True)

KEEP = 4000  # сколько топ-кандидатов храним по каждому каналу
N_FEAT = 7  # bmF, bmT, dense, transfer, microcat, title_all, popularity
N_TOTAL = N_FEAT + 5  # # + exact, loc, rating, cov(query), cov(query+infm)


def rank_score(n):
    # Оценка кандидата по рангу в канале: 1-й -> 1.0, последний -> ~0.
    return 1.0 - np.arange(n) / max(n, 1)


def make_coverage(bm, token_index):
    # Фабрика признака «покрытия запроса»: какая доля суммарного IDF термов
    # запроса реально встречается в полном тексте объявления. Использует только
    # текст запроса и корпус — утечки через клики нет.
    mat = bm.mat
    idf = bm.idf

    def cov(ids, text):
        qterms = [token_index.get(t) for t in str(text).split()]
        qterms = sorted(set(int(c) for c in qterms if c is not None))
        n = len(ids)
        if not qterms or n == 0:
            return np.zeros(n, dtype=np.float32)
        wsum = float(np.sum(idf[qterms]))
        cur = np.zeros(n, dtype=np.float32)
        for c in qterms:
            s, e = mat.indptr[c], mat.indptr[c + 1]
            if s == e:
                continue
            hit = np.isin(ids, mat.indices[s:e])
            cur[hit] += idf[c] if wsum > 0 else 1.0
        return cur / wsum

    return cov


def build_feature_matrix(res, qi, loc, rate, cov_fn=None, df=None):
    # Объединяем кандидатов всех каналов в единую матрицу признаков
    # (score канала = 1 - ранг/длина, чтобы каналы были сопоставимы),
    # затем дополняем бинарными признаками exact / loc / rating и покрытия запроса.
    data = {}
    ex_set = set(int(x) for x in res["exact_ids"][qi])

    def store(key, ids, sc):
        for j, ii in enumerate(ids):
            d = data.get(int(ii))
            if d is None:
                d = np.zeros(N_FEAT, dtype=np.float32)
                data[int(ii)] = d
            d[key] = sc[j]

    store(0, res["bmF_ids"][qi], rank_score(len(res["bmF_ids"][qi])))
    store(1, res["bmT_ids"][qi], rank_score(len(res["bmT_ids"][qi])))
    store(2, res["dense_ids"][qi], rank_score(len(res["dense_ids"][qi])))
    store(3, res["trans_ids"][qi], rank_score(len(res["trans_ids"][qi])))
    store(4, res["mc_ids"][qi], rank_score(len(res["mc_ids"][qi])))
    store(5, res["tall_ids"][qi], rank_score(len(res["tall_ids"][qi])))
    store(6, res["pop_ids"][qi], rank_score(len(res["pop_ids"][qi])))
    if not data:
        return np.empty(0, dtype=np.int64), np.empty((0, N_FEAT), dtype=np.float32)
    ids = np.fromiter(data.keys(), dtype=np.int64)
    X = np.stack([data[i] for i in ids.tolist()])
    ex_flag = np.isin(ids, list(ex_set)).astype(np.float32)
    X = np.concatenate([X, ex_flag[:, None],
                        loc[ids].astype(np.float32)[:, None],
                        rate[ids].astype(np.float32)[:, None]], axis=1)
    if cov_fn is not None and df is not None:
        cov1 = cov_fn(ids, str(df.loc[qi, "s_query"]))
        cov2 = cov_fn(ids, str(df.loc[qi, "s_query"]) + " " + str(df.loc[qi, "s_infm"]))
        X = np.concatenate([X, cov1[:, None], cov2[:, None]], axis=1)
    return ids, X


class BM25:
    def __init__(self, mat, doc_len, k1=1.5, b=0.75):
        self.mat = mat.tocsc()
        self.doc_len = doc_len.astype(np.float64)
        n = self.mat.shape[0]
        # Частота документа (df) вычисляется из indptr: сколько записей в колонке-терме.
        df = np.diff(self.mat.indptr).clip(min=1)
        self.idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float64)
        self.avgdl = doc_len.mean()
        self.k1 = k1
        self.b = b
        self.N = n

    def score(self, q_cols):
        q_cols = sorted(set(int(c) for c in q_cols if c is not None))
        if not q_cols:
            return np.zeros(self.N, dtype=np.float64)
        # Быстрый вариант BM25: идём по postings-спискам термов запроса и набираем
        # вес только для документов, где термин встречается (вместо плотной N x |q|).
        mat = self.mat
        acc = np.zeros(self.N, dtype=np.float64)
        for c in q_cols:
            s, e = mat.indptr[c], mat.indptr[c + 1]
            if s == e:
                continue
            idx = mat.indices[s:e]
            f = mat.data[s:e].astype(np.float64)
            dl = self.doc_len[idx]
            base = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            contrib = f * (self.k1 + 1) * self.idf[c] / (f + base)
            np.add.at(acc, idx, contrib)
        return acc


class Retriever:
    def __init__(self):
        print("Loading caches...", flush=True)
        t0 = time.time()
        self.items = pd.read_parquet(CACHE + r"\items.parquet")
        self.tr = pd.read_parquet(CACHE + r"\train.parquet")
        self.tq = pd.read_parquet(CACHE + r"\train_queries.parquet")
        with open(CACHE + r"\token_index.pkl", "rb") as f:
            self.token_index = pickle.load(f)
        full_mat = sp.load_npz(CACHE + r"\full_mat.npz")
        title_mat = sp.load_npz(CACHE + r"\title_mat.npz")
        doc_len = np.load(CACHE + r"\doc_len.npy")
        t_doc_len = np.load(CACHE + r"\titel_doc_len.npy")
        self.full_bm = BM25(full_mat, doc_len, k1=1.5, b=0.75)
        self.title_bm = BM25(title_mat, t_doc_len, k1=1.3, b=0.5)
        self._title_csr = title_mat
        del full_mat
        self.item_emb = np.load(CACHE + r"\item_emb.npy").astype(np.float32)
        self.bq_emb = np.load(CACHE + r"\bench_q_emb.npy").astype(np.float32)
        self.trq_emb = np.load(CACHE + r"\train_q_emb.npy").astype(np.float32)
        self.item_emb = sknorm(self.item_emb)
        self.bq_emb = sknorm(self.bq_emb)
        self.trq_emb = sknorm(self.trq_emb)
        # Популярность: сколько раз каждый товар корпуса кликнули в train.
        self._i2pos = {v: i for i, v in enumerate(self.items["item_id"])}
        self.item_pop = self._compute_pop(set())
        self._pop_rank = np.argsort(-self.item_pop)[:KEEP]  # топ самых кликаемых товаров корпуса
        # Канал mc: микрокатегории кликнутых товаров (разреженная матрица запрос x микрокатегория).
        self._build_clicked_micro()
        # Для каждой микрокатегории — список позиций объявлений корпуса в этой категории.
        self._corpus_micro_items = self.items.groupby("item_microcat_id").apply(
            lambda g: g.index.values).to_dict()
        # Сопоставление «запрос -> номер строки» в матрице микрокатегорий
        # (нужно, чтобы обнулять строки исключённых валидационных запросов).
        self._qrow = {q: i for i, q in enumerate(self.tq["n_q"])}
        print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    def _compute_pop(self, exclude):
        # Счёт кликов по товарам корпуса; exclude отбрасывает валидационные запросы.
        tr = self.tr if not exclude else self.tr[~self.tr["n_query"].isin(exclude)]
        pop = np.zeros(len(self.items), dtype=np.int32)
        for item_id, c in tr["item_id"].value_counts().items():
            p = self._i2pos.get(item_id)
            if p is not None:
                pop[p] = c
        return pop

    def _build_clicked_micro(self):
        cpath = CACHE + r"\clicked_mc.npz"
        mcpath = CACHE + r"\mc_ids.npy"
        if os.path.exists(cpath) and os.path.exists(mcpath):
            self._clicked_mc = sp.load_npz(cpath)
            self._mc_ids = np.load(mcpath)
            return
        # Разреженная матрица «train-запрос x микрокатегория» = число кликов.
        # Через неё канал mc находит микрокатегории, релевантные похожим запросам.
        full_tr = pd.read_parquet(os.path.join(BASE, "train.parquet"),
                                  columns=["search_query", "item_id", "item_microcat_id"])
        full_tr = full_tr.merge(self.tq[["search_query", "n_q"]].rename(columns={"n_q": "n_query"}),
                                on="search_query", how="inner")
        full_tr["n_query"] = full_tr["n_query"].fillna("")
        g = full_tr.groupby(["n_query", "item_microcat_id"]).size().reset_index(name="n")
        q2pos = {q: i for i, q in enumerate(self.tq["n_q"])}
        rows = g["n_query"].map(q2pos).values
        cols = g["item_microcat_id"].values
        uniq_mc = np.unique(cols)
        self._mc_ids = uniq_mc
        mc2c = {m: i for i, m in enumerate(uniq_mc)}
        cols_c = np.array([mc2c[m] for m in cols])
        self._clicked_mc = sp.csr_matrix((g["n"].values.astype(np.float32), (rows, cols_c)),
                                         shape=(len(self.tq), len(uniq_mc))).tocsr()
        sp.save_npz(cpath, self._clicked_mc)
        np.save(mcpath, self._mc_ids)

    def build_q_click(self, exclude=None):
        # Для каждого train-запроса — позиции кликнутых товаров, присутствующих в корпусе.
        # exclude убирает валидационные запросы, чтобы не было утечки через канал exact.
        exclude = exclude or set()
        q_map = {}
        for q, g in self.tr.groupby("n_query"):
            if q in exclude:
                continue
            idx = np.array([self._i2pos[x] for x in g["item_id"] if x in self._i2pos],
                           dtype=np.int64)
            if len(idx):
                q_map[q] = idx
        return q_map

    def set_q_map(self, exclude=None):
        exclude = exclude or set()
        self._q_map = self.build_q_click(exclude)
        if exclude:
            # Анти-утечка для валидации: исключённые запросы убираем и из каналов,
            # которые опираются на глобальную статистику train (популярность, микрокатегории).
            # Без этого свои же клики валидационного запроса поднимали бы его кандидатов.
            self.item_pop = self._compute_pop(exclude)
            self._pop_rank = np.argsort(-self.item_pop)[:KEEP]
            m = self._clicked_mc.tolil()
            for q in exclude:
                r = self._qrow.get(q)
                if r is not None:
                    m.rows[r] = []
                    m.data[r] = []
            self._clicked_mc = m.tocsr()
        return self._q_map

    def retrieve(self, df, n_sim=90, sim_gate=0.30, chunk=250, n_microcats=6):
        nq = len(df)
        out = {k: [] for k in
               ["bmF_ids", "bmF_scores", "bmT_ids", "bmT_scores",
                "dense_ids", "dense_scores", "trans_ids", "trans_scores", "exact_ids",
                "mc_ids", "mc_scores", "tall_ids", "tall_scores", "pop_ids", "pop_scores"]}
        qcols = []
        qtok_all = []
        # Разбираем токены (стемы) запроса один раз и переиспользуем во всех каналах.
        for _, r in df.iterrows():
            qtokens = str(r["s_query"]).split() + str(r["s_infm"]).split()
            qcols.append([self.token_index.get(t) for t in qtokens])
            qtok_all.append(str(r["s_query"]).split())

        print("BM25...", flush=True)
        N = self.full_bm.N
        for i in range(nq):
            # BM25-оценки по полному тексту и по заголовку; берём топ-K позиций.
            fs = self.full_bm.score(qcols[i])
            ts = self.title_bm.score(qcols[i])
            o = np.argpartition(-fs, min(KEEP, N - 1))[:KEEP]
            o = o[np.argsort(-fs[o])]
            out["bmF_ids"].append(o)
            out["bmF_scores"].append(fs[o].astype(np.float32))
            o = np.argpartition(-ts, min(KEEP, N - 1))[:KEEP]
            o = o[np.argsort(-ts[o])]
            out["bmT_ids"].append(o)
            out["bmT_scores"].append(ts[o].astype(np.float32))
            # ---- title_all: объявления, где в заголовке есть все слова запроса ----
            # (жёсткое условие вместо мягкого BM25-ранжирования)
            tok_ids = sorted(set(self.token_index.get(t) for t in qtok_all[i] if self.token_index.get(t) is not None))
            if tok_ids:
                sub = self._title_csr[:, tok_ids]
                row_ok = np.asarray(sub.sum(axis=1)).ravel() == len(tok_ids)
                idx_t = np.flatnonzero(row_ok)
                if len(idx_t):
                    o2 = idx_t[np.argsort(-ts[idx_t])][:KEEP]
                    out["tall_ids"].append(o2)
                    out["tall_scores"].append(ts[o2].astype(np.float32))
                else:
                    out["tall_ids"].append(np.empty(0, dtype=np.int64))
                    out["tall_scores"].append(np.empty(0, dtype=np.float32))
            else:
                out["tall_ids"].append(np.empty(0, dtype=np.int64))
                out["tall_scores"].append(np.empty(0, dtype=np.float32))
            if (i + 1) % 400 == 0:
                print(f"  bm {i+1}/{nq}", flush=True)

        # ---- Dense-канал: матричное умножение по частям, чтобы не строить всю
        # матрицу близостей (nq x 189k) в памяти сразу.
        print("Dense...", flush=True)
        qemb = sknorm(np.stack(list(df["emb"])))
        pop_ids = self._pop_rank
        pop_scores = (1.0 - np.arange(len(pop_ids)) / len(pop_ids)).astype(np.float32)
        for st in range(0, nq, chunk):
            en = min(st + chunk, nq)
            sim = qemb[st:en] @ self.item_emb.T
            for j in range(st, en):
                row = sim[j - st]
                o = np.argpartition(-row, min(KEEP, N - 1))[:KEEP]
                o = o[np.argsort(-row[o])]
                out["dense_ids"].append(o)
                out["dense_scores"].append(row[o].astype(np.float32))
            del sim
        # ---- Transfer: перенос кликов ----
        # Берём топ похожих train-запросов и собираем их кликнутые объявления.
        print("Transfer...", flush=True)
        qs = self.trq_emb @ qemb.T
        nn_q = df["n_query"].tolist()
        # ---- Microcat-канал ----
        # mc_scores_q[i] — вес каждой микрокатегории для i-го запроса.
        print("Microcat...", flush=True)
        mc_scores_q = qs.T @ self._clicked_mc  # (nq, число микрокатегорий), вес = близость запросов
        for i in range(nq):
            # Для каждого кандидата оставляем максимальную близость среди похожих запросов.
            cand = {}
            o = np.argsort(-qs[:, i])[:n_sim]
            for s in o:
                simv = float(qs[s, i])
                if simv < sim_gate:
                    continue
                idxs = self._q_map.get(self.tq.iloc[s]["n_q"])
                if idxs is None:
                    continue
                for ix in idxs:
                    ii = int(ix)
                    old = cand.get(ii)
                    if old is None or simv > old:
                        cand[ii] = simv
            items = np.fromiter(cand.keys(), dtype=np.int64)
            scrs = np.fromiter(cand.values(), dtype=np.float32)
            if len(items):
                ordr = np.argsort(-scrs)
                items, scrs = items[ordr], scrs[ordr]
            out["trans_ids"].append(items)
            out["trans_scores"].append(scrs)
            # exact: клики по точно такому же тексту запроса в train — самые надёжные кандидаты.
            ex = self._q_map.get(nn_q[i])
            out["exact_ids"].append(ex if (ex is not None and ex.size) else np.empty(0, dtype=np.int64))
            # ---- microcat-канал ----
            # Берём топ микрокатегорий из кликов похожих запросов и вытягиваем их товары.
            mrow = mc_scores_q[i]
            mc_top = np.argsort(-mrow)[:n_microcats]
            parts = []
            for cix in mc_top:
                mcv = float(mrow[cix])
                if mcv <= 0:
                    continue
                real_mc = self._mc_ids[cix]
                got = self._corpus_micro_items.get(real_mc)
                if got is None:
                    continue
                # Кап на размер микрокатегории; перемешивание (interleave) не даёт одной
                # категории вытеснить остальные.
                if len(got) > 300:
                    got = got[:300]
                parts.append((mcv, got))
            if parts:
                parts.sort(key=lambda p: -p[0])
                inter = []
                mx = max(len(g) for _, g in parts)
                for j in range(mx):
                    for _, g in parts:
                        if j < len(g):
                            inter.append(g[j])
                allids = np.asarray(inter, dtype=np.int64)[:KEEP]
                # "Сила" кандидата ~ линейное убывание с позицией в канале.
                strength = (1.0 - np.arange(len(allids)) / len(allids)).astype(np.float32)
                out["mc_ids"].append(allids)
                out["mc_scores"].append(strength)
            else:
                out["mc_ids"].append(np.empty(0, dtype=np.int64))
                out["mc_scores"].append(np.empty(0, dtype=np.float32))
            if (i + 1) % 400 == 0:
                print(f"  trans/mc {i+1}/{nq}", flush=True)
            out["pop_ids"].append(pop_ids)
            out["pop_scores"].append(pop_scores)

        final = {}
        for key in out:
            # Каналы имеют разную длину на запрос (например, title_all может быть
            # пустым), поэтому храним списки как object-массивы без выравнивания.
            arr = np.array(out[key], dtype=object)
            final[key] = arr
        return final


if __name__ == "__main__":
    r = Retriever()
    r.set_q_map()
    bq = pd.read_parquet(CACHE + r"\benchmark_queries.parquet")
    bq["emb"] = r.bq_emb
    res = r.retrieve(bq)
    with open(OUT + r"\bench_retrieval.pkl", "wb") as f:
        pickle.dump(res, f)
    print("bench retrieval saved.")
    print({k: (v.shape if hasattr(v, "shape") else "obj") for k, v in res.items()})