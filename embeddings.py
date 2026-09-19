# -*- coding: utf-8 -*-
"""
Этап 2: эмбеддинги.
Модель cointegrated/rubert-tiny2 — маленький русскоязычный би-энкодер,
достаточно быстрый на CPU. Кэшируем векторы (312-мерные):
  item_emb.npy — объявления, bench_q_emb.npy — запросы бенчмарка,
  train_q_emb.npy — запросы train (тот же порядок, что в train_queries.parquet).
"""
import sys, time, torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(12)

from config import CACHE

MODEL = "cointegrated/rubert-tiny2"


def truncate(texts, n_tokens=256):
    # Обрезаем длинные описания до n токенов — модель имеет фиксированное окно внимания.
    out = []
    for t in texts:
        if t is None:
            out.append("")
            continue
        parts = str(t).split()
        out.append(" ".join(parts[:n_tokens]))
    return out


def encode_big(model, texts, batch=128, desc=""):
    # Кодирование большими батчами: по одному батчу за раз, чтобы не переполнить RAM.
    embs = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        e = model.encode(chunk, batch_size=batch, show_progress_bar=False, normalize_embeddings=True)
        embs[i:i + len(chunk)] = e
        if (i // batch) % 25 == 0:
            spd = (i + len(chunk)) / (time.time() - t0)
            print(f"  {desc}: {i + len(chunk)}/{len(texts)}  {spd:.1f} it/s", flush=True)
    print(f"  {desc}: done {len(texts)} in {time.time()-t0:.1f}s", flush=True)
    return embs


def main():
    print("Loading caches...")
    items = pd.read_parquet(CACHE + r"\items.parquet")
    bq = pd.read_parquet(CACHE + r"\benchmark_queries.parquet")
    tr = pd.read_parquet(CACHE + r"\train.parquet")

    # --- строим упорядоченный список уникальных запросов train ---
    qdf = tr.groupby("n_query", sort=False).agg(
        s_q=("s_query", "first"),
        n_q=("n_query", "first")
    ).reset_index()
    qdf = qdf.rename(columns={"n_query": "search_query"})
    print("distinct train queries:", len(qdf))
    qdf[["search_query", "s_q", "n_q"]].to_parquet(CACHE + r"\train_queries.parquet")

    print("Loading model", MODEL)
    model = SentenceTransformer(MODEL)

    # ---- объявления ----
    # Склеиваем все поля объявления в одну строку: заголовок | описание | параметры.
    comb_item = (items["n_title"] + " | " + items["n_desc"] + " | " + items["n_infm"]).tolist()
    comb_item = truncate(comb_item, 256)
    item_emb = encode_big(model, comb_item, batch=128, desc="items")
    np.save(CACHE + r"\item_emb.npy", item_emb)

    # ---- запросы бенчмарка ----
    # В запрос добавляем текст параметров фильтра поиска — он уточняет намерение.
    comb_bq = (bq["n_query"] + " | " + bq["n_infm"]).tolist()
    comb_bq = truncate(comb_bq, 256)
    bq_emb = encode_big(model, comb_bq, batch=128, desc="bench queries")
    np.save(CACHE + r"\bench_q_emb.npy", bq_emb)

    # ---- запросы train ----
    # Кодируем в том же формате (текст запроса), чтобы их можно было сравнивать
    # с запросами бенчмарка для канала «перенос кликов».
    comb_tr = (qdf["n_q"] + " | ").tolist()
    comb_tr = truncate(comb_tr, 128)
    trq_emb = encode_big(model, comb_tr, batch=128, desc="train queries")
    np.save(CACHE + r"\train_q_emb.npy", trq_emb)

    print("embeddings_saved")


if __name__ == "__main__":
    main()