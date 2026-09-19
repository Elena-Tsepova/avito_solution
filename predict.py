# -*- coding: utf-8 -*-
"""
Этап 5: инференс на бенчмарке -> out/answer.csv.
Для 2452 запросов запускаем все каналы поиска, считаем линейный балл
кандидатов с весами из validate.py (out/best_weights.npy), берём топ-50
объявлений и записываем ответ.
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
from config import CACHE, OUT, HERE
sys.path.insert(0, HERE)
from retrieval import (Retriever, build_feature_matrix, make_coverage,
                       N_TOTAL)

TOP = 50


def main():
    r = Retriever()
    app_cov = make_coverage(r.full_bm, r.token_index)
    bq = pd.read_parquet(CACHE + r"\benchmark_queries.parquet").reset_index(drop=True)
    bq["emb"] = list(r.bq_emb)

    res_path = OUT + r"\bench_retrieval.pkl"
    # Кандидаты по всем каналам на бенчмарке дорогие — кэшируем их между попытками.
    if os.path.exists(res_path):
        print("Loading cached benchmark retrieval...")
        with open(res_path, "rb") as f:
            res = pickle.load(f)
    else:
        r.set_q_map(exclude=set())
        res = r.retrieve(bq, n_sim=90, sim_gate=0.30, n_microcats=8)
        with open(res_path, "wb") as f:
            pickle.dump(res, f)

    # Разные веса для двух групп запросов:
    # - текст запроса есть в train (overlap) -> веса, где свой клик в train важнее всего;
    # - остальных (fresh) -> веса из честной валидации без утечки.
    w_default = np.array([2.2, 0.02, 0.0, 0.22, 0.0, 0.57, 0.25, 3.0, 0.6, 0.4,
                          0.3, 0.2], dtype=np.float32)
    w = np.load(OUT + r"\best_weights.npy").astype(np.float32) \
        if os.path.exists(OUT + r"\best_weights.npy") else w_default
    w_overlap = np.load(OUT + r"\best_weights_overlap.npy").astype(np.float32) \
        if os.path.exists(OUT + r"\best_weights_overlap.npy") else w
    train_nq = set(pd.read_parquet(CACHE + r"\train_queries.parquet")["n_q"].tolist())
    n_overlap = bq["n_query"].isin(train_nq).sum()
    print("Using fresh weights:", w.tolist())
    print("Using overlap weights:", w_overlap.tolist())
    print("overlap queries (in train):", n_overlap, "fresh:", len(bq) - n_overlap)

    item_ids = r.items["item_id"].values            # item_id — строки (16-значный hex)
    item_locs = r.items["item_location_id"].values
    # Фильтр «Рейтинг пользователя 4 звезды и выше»: такие объявления поднимаем выше.
    rating_flags = bq["n_infm"].fillna("").str.contains("рейтинг", na=False).values
    rate_arr = (r.items["item_rating"].fillna(0).values >= 4.0)
    n_items = len(item_ids)

    rows = []
    for qi in range(len(bq)):
        w = w_overlap if bq.loc[qi, "n_query"] in train_nq else w
        s_loc = bq.loc[qi, "search_location_id"]
        # Совпадение локации — сильный сигнал (в 83% кликов локация совпадает с поиском).
        loc = (item_locs == s_loc) if pd.notna(s_loc) else np.zeros(n_items, dtype=bool)
        rate = rate_arr if rating_flags[qi] else np.zeros(n_items, dtype=bool)
        ids, X = build_feature_matrix(res, qi, loc, rate, cov_fn=app_cov, df=bq)
        if len(ids) == 0:
            rows.append((bq.loc[qi, "query_id"], ""))
            continue
        # Сортируем кандидатов по линейному баллу и берём топ-50.
        sc = X @ w
        ntop = min(TOP, len(ids))
        top_ix = np.argpartition(-sc, ntop - 1)[:ntop]
        ord_ = np.argsort(-sc[top_ix])
        ids50 = ids[top_ix[ord_]]
        ans = " ".join(item_ids[i] for i in ids50)
        rows.append((bq.loc[qi, "query_id"], ans))

    out = pd.DataFrame(rows, columns=["query_id", "answer"])
    out_path = OUT + r"\answer.csv"
    # Ответ: query_id и колонка answer (item_id через пробел, до 50 штук).
    out.to_csv(out_path, index=False)
    print("Wrote", out_path, "rows:", len(out), flush=True)
    print("Queries:", len(out), "max items per answer:",
          out["answer"].str.split().str.len().max())


if __name__ == "__main__":
    main()