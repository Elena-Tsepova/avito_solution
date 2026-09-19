# -*- coding: utf-8 -*-
"""
Этап 4: валидация и подбор весов каналов.
На отдельной выборке из train (эти запросы исключаются и из канала transfer,
и из матрицы микрокатегорий, и из популярности — чтобы не было утечки)
считаем Recall@50 и подбираем веса признаков.

Признаки на объявление-кандидата (0..1): 7 базовых каналов (bmF, bmT, dense,
transfer, microcat, title_all, popularity) + 3 бинарных (exact, loc, rating).
Итоговый балл = w @ x; ответ — топ-50 по баллу.
"""
import sys, os, pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(6)

from config import CACHE, OUT, HERE
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from retrieval import (Retriever, build_feature_matrix, make_coverage,
                       N_FEAT, N_TOTAL)

TOP = 50


def loc_by_q():
    # Сопоставление «запрос -> локация поиска», берётся из кэша train.
    m = pd.read_parquet(CACHE + r"\train.parquet")[["n_query", "search_location_id"]]
    return m.drop_duplicates("n_query").set_index("n_query")["search_location_id"].to_dict()


def build_val(tr, r, n_val=1600, seed=42):
    # Истинные ответы = кликнутые в train товары, которые есть в корпусе.
    truth = {}
    for q, g in tr.groupby("n_query"):
        idx = np.array([r.pos.get(x, -1) for x in g["item_id"]], dtype=np.int64)
        idx = idx[idx >= 0]
        if len(idx):
            truth[q] = set(idx.tolist())
    qs = list(truth.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(qs)
    sel = qs[:n_val]
    df = tr[tr["n_query"].isin(sel)][["n_query", "s_query", "s_infm", "n_infm"]].drop_duplicates("n_query")
    truth_sel = {q: truth[q] for q in sel}
    return df, truth_sel


class Validator:
    def __init__(self, res, df, truth, item_locs, s_locs, rating_flags, item_rate=None,
                 feats=None, cov_fn=None):
        if item_rate is None:
            item_rate = np.zeros(len(item_locs), dtype=bool)
        if feats is not None:
            self.feats = feats
            return
        self.feats = []
        for qi in range(len(df)):
            s_loc = s_locs[qi]
            loc = (item_locs == s_loc) if pd.notna(s_loc) else np.zeros(len(item_locs), dtype=bool)
            rate = np.zeros(len(item_locs), dtype=bool) if not rating_flags[qi] else item_rate
            ids, X = build_feature_matrix(res, qi, loc, rate,
                                          cov_fn=cov_fn, df=df)
            truths_ = np.zeros(len(ids), dtype=bool)
            tq = truth[df["n_query"].iloc[qi]]
            pos_map = {int(i): j for j, i in enumerate(ids)}
            for t in tq:
                if t in pos_map:
                    truths_[pos_map[t]] = True
            self.feats.append((ids, X, truths_))

    def eval(self, w):
        w = np.asarray(w, dtype=np.float32)
        total_hit = 0.0
        # Векторизованная оценка: балл = X @ w, топ-50 через argpartition.
        for ids, X, truths_ in self.feats:
            if len(ids) == 0:
                continue
            sc = X @ w
            ntop = min(TOP, len(ids))
            top_ix = np.argpartition(-sc, ntop - 1)[:ntop]
            hit = int(truths_[top_ix].sum())
            total_hit += hit / max(1, int(truths_.sum()))
        return total_hit / len(self.feats)

    def greedy(self, n_iter=4, step=0.35, base=None):
        # Простой координатный спуск по весам: меняем по одному признаку,
        # остальные фиксированы; несколько проходов до стабилизации.
        if base is None:
            base = np.array([1.0, 1.0, 0.7, 1.0, 0.4, 0.4, 0.25, 3.0, 0.6, 0.4,
                             0.3, 0.2], dtype=np.float32)
        best_w = base.copy()
        best_r = self.eval(best_w)
        cands = [-0.6, -0.35, 0.0, 0.35, 0.6]
        for it in range(n_iter):
            improved = False
            for f in range(N_FEAT):
                cur = best_w[f]
                scores = []
                for d in cands:
                    v = max(0.0, cur + d)
                    w = best_w.copy()
                    w[f] = v
                    scores.append((self.eval(w), v))
                scores.sort(key=lambda x: -x[0])
                if scores[0][0] > best_r + 1e-6:
                    best_w[f] = scores[0][1]
                    best_r = scores[0][0]
                    improved = True
                    print(f"  iter {it} feat {f}: R@50={best_r:.4f}  w[{f}]={best_w[f]:.2f}", flush=True)
            if not improved:
                break
        return best_w, best_r


def fine_search(v):
    # Дробление шага после грубого greedy: пока одна координата не перестанет
    # улучшать метрику. Стартуем с весов, найденных greedy.
    w = np.array([2.2, 0.02, 0.0, 0.22, 0.0, 0.42, 0.25, 3.0, 0.6, 0.4,
                  0.3, 0.2], dtype=np.float32)
    best = v.eval(w)
    for step in (0.1, 0.05, 0.02):
        for _ in range(6):
            improved = False
            for f in range(N_TOTAL):
                cur = best
                for d in (-step, step):
                    w2 = w.copy()
                    w2[f] = max(0.0, w2[f] + d)
                    s = v.eval(w2)
                    if s > cur + 1e-6:
                        cur = s
                        w = w2
                if cur > best + 1e-6:
                    best = cur
                    improved = True
            if not improved:
                break
    return w, best


def main():
    mode = "overlap" if len(sys.argv) > 1 and sys.argv[1] == "overlap" else "fresh"
    r = Retriever()
    r.pos = {v: i for i, v in enumerate(r.items["item_id"])}
    item_locs = r.items["item_location_id"].values
    item_rate = (r.items["item_rating"].fillna(0).values >= 4.0)

    feats_path = (OUT + r"\val_feats_overlap.pkl" if mode == "overlap"
                  else OUT + r"\val_feats.pkl")
    if os.path.exists(feats_path):
        print("Loading cached validation features...")
        with open(feats_path, "rb") as f:
            df, truth, feats = pickle.load(f)
    else:
        df, truth = build_val(r.tr, r, n_val=1600)
        print("val queries:", len(df))
        emb_cache = OUT + r"\val_emb.npy"
        if os.path.exists(emb_cache):
            print("Loading cached val embeddings...")
            emb = np.load(emb_cache)
        else:
            print("Embedding val queries...")
            model = SentenceTransformer("cointegrated/rubert-tiny2")
            texts = (df["n_query"] + " | " + df["n_infm"].fillna("")).tolist()
            emb = model.encode(texts, batch_size=128, show_progress_bar=False,
                               normalize_embeddings=True)
            np.save(emb_cache, emb)
        df = df.reset_index(drop=True)
        df["emb"] = list(emb)
        # overlap-режим: утечка включена намеренно — здесь она честно имитирует
        # запросы, чей текст уже есть в train (свои клики в train доступны).
        # Для fresh-режима утечка по-прежнему выключается.
        r.set_q_map(exclude=set() if mode == "overlap" else set(df["n_query"]))
        res = r.retrieve(df, n_sim=90, sim_gate=0.30, n_microcats=8)
        s_locs = df["n_query"].map(loc_by_q()).values
        rating_flags = df["n_infm"].fillna("").str.contains("рейтинг", na=False).values
        v = Validator(res, df, truth, item_locs, s_locs, rating_flags, item_rate=item_rate)
        feats = v.feats
        with open(feats_path, "wb") as f:
            pickle.dump((df, truth, feats), f)

    print("Rebuilding validator...")
    s_locs = df["n_query"].map(loc_by_q()).values
    rating_flags = df["n_infm"].fillna("").str.contains("рейтинг", na=False).values
    v = Validator(None, df, truth, item_locs, s_locs, rating_flags,
                  item_rate=item_rate, feats=feats)

    # Оценка каналов по отдельности — понимаем, что даёт каждый признак сам по себе.
    for fi, name in enumerate(["BM25F", "BM25T", "Dense", "Transfer", "Microcat",
                               "TitleAll", "Pop", "Exact", "Loc", "Rating"]):
        w = np.zeros(N_TOTAL, dtype=np.float32)
        w[fi] = 1.0
        print(f"Solo {name}: R@50 = {v.eval(w):.4f}", flush=True)

    best_w, best_r = fine_search(v)
    wout = (OUT + r"\best_weights_overlap.npy" if mode == "overlap"
            else OUT + r"\best_weights.npy")
    np.save(wout, best_w)  # эти веса читает predict.py
    print("BEST WEIGHTS:", best_w.tolist())
    print("BEST VAL R@50:", best_r)

    # Несколько прогонов — желательно ещё подобрать веса для второй группы запросов.
    # Готовим веса для overlap на уже построенном кэше (если он есть).
    if mode != "overlap" and os.path.exists(OUT + r"\val_feats_overlap.pkl"):
        with open(OUT + r"\val_feats_overlap.pkl", "rb") as f:
            df2, truth2, feats2 = pickle.load(f)
        s_locs2 = df2["n_query"].map(loc_by_q()).values
        rating2 = df2["n_infm"].fillna("").str.contains("рейтинг", na=False).values
        v2 = Validator(None, df2, truth2, item_locs, s_locs2, rating2,
                       item_rate=item_rate, feats=feats2)
        w2, r2 = fine_search(v2)
        np.save(OUT + r"\best_weights_overlap.npy", w2)
        print("OVERLAP BEST WEIGHTS:", w2.tolist())
        print("OVERLAP BEST VAL R@50:", r2)


if __name__ == "__main__":
    main()