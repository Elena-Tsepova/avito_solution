# -*- coding: utf-8 -*-
"""
Этап 1: предобработка данных.
Нормализация текста, стемминг pymorphy2, построение разреженных матриц
для BM25 (по полному тексту и по заголовку), сохранение кэшей в ./cache.
"""
import re
import sys
import inspect
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

if not hasattr(inspect, "getargspec"):  # обходной приём: pymorphy2 несовместим с Python 3.11
    inspect.getargspec = lambda f: inspect.getfullargspec(f)[:4]
from pymorphy2 import MorphAnalyzer

sys.stdout.reconfigure(encoding='utf-8')

# Пути берутся относительно config.py
from config import BASE, CACHE

PUNCT_RE = re.compile(r"[^a-zа-яё0-9]+")


def normalize(text: str) -> str:
    # Нормализация: нижний регистр, замена пунктуации на пробелы, схлопывание пробелов.
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    print("Loading data...")
    tr = pd.read_parquet(BASE + r"\train.parquet")
    bq = pd.read_parquet(BASE + r"\benchmark_queries.parquet")
    bi = pd.read_parquet(BASE + r"\benchmark_items.parquet")

    # --- Нормализуем текстовые поля: объявления, запросы, параметры фильтров ---
    bi["n_title"] = bi["item_title_raw"].fillna("").map(normalize)
    bi["n_desc"] = bi["item_description_raw"].fillna("").map(normalize)
    bi["n_infm"] = bi["item_infm_params_text"].fillna("").map(normalize)
    bi["fulltext"] = bi["n_title"] + " " + bi["n_desc"] + " " + bi["n_infm"]

    tr["n_query"] = tr["search_query"].fillna("").map(normalize)
    tr["n_infm"] = tr["search_infm_params_text"].fillna("").map(normalize)
    bq["n_query"] = bq["search_query"].fillna("").map(normalize)
    bq["n_infm"] = bq["search_infm_params_text"].fillna("").map(normalize)

    # Стемминг
    # Приводим слова к начальной форме, чтобы формы «окон» и «окна» совпадали.
    print("Collecting tokens...")
    morph = MorphAnalyzer()

    all_tokens = set()
    for txt in bi["fulltext"].tolist():
        all_tokens.update(txt.split())
    for txt in tr["n_query"].tolist():
        all_tokens.update(txt.split())
    for txt in bq["n_query"].tolist():
        all_tokens.update(txt.split())
    for txt in tr["n_infm"].tolist():
        all_tokens.update(txt.split())
    for txt in bq["n_infm"].tolist():
        all_tokens.update(txt.split())
    # Сортировка делает стемминг воспроизводимым между запусками.
    all_tokens = sorted(all_tokens)
    print("Vocabulary size:", len(all_tokens))

    print("Stemming vocabulary...")
    stem = {}
    for i, tok in enumerate(all_tokens):
        p = morph.parse(tok)[0]
        stem[tok] = p.normal_form if p.normal_form else tok
        if (i + 1) % 20000 == 0:
            print("  stemmed", i + 1)

    # Применяем стемминг к текстам объявлений и запросов.
    def st(text):
        return " ".join(stem.get(t, t) for t in text.split())

    def st_list(text):
        return [stem.get(t, t) for t in text.split()]

    print("Stemming item texts...")
    # Для разреженных матриц храним списки стемов в виде numpy object-массивов.
    full_lists = bi["fulltext"].map(st_list).tolist()
    title_lists = bi["n_title"].map(st_list).tolist()
    # Стем-заголовки отдельной колонкой: их используют как источник терминов
    # расширения запроса (релеванс-фидбек по кликнутым объявлениям похожих запросов).
    bi["s_title"] = bi["n_title"].map(st)

    tr["s_query"] = tr["n_query"].map(st)
    tr["s_infm"] = tr["n_infm"].map(st)
    bq["s_query"] = bq["n_query"].map(st)
    bq["s_infm"] = bq["n_infm"].map(st)

    # ---- Построение словаря стемов и разреженных матриц ----
    # Индекс «токен -> номер колонки» для матриц документ x токен.
    token_index = {}
    for lst in full_lists:
        for t in lst:
            if t not in token_index:
                token_index[t] = len(token_index)
    print("Stem token vocab size:", len(token_index))

    # Матрица «документ x токен»: 1 — если токен встречается в документе.
    print("Building sparse matrices...")
    def build_csr(lists, vocab):
        rows, cols, data = [], [], []
        for di, lst in enumerate(lists):
            for t in lst:
                ci = vocab.get(t)
                if ci is not None:
                    rows.append(di)
                    cols.append(ci)
                    data.append(1)
        m = sp.csr_matrix((data, (rows, cols)), shape=(len(lists), len(vocab)), dtype=np.float32)
        m.sum_duplicates()
        return m

    full_mat = build_csr(full_lists, token_index)
    title_mat = build_csr(title_lists, token_index)
    full_mat = full_mat.tocsc()
    title_mat = title_mat.tocsc()
    print("full_mat nnz:", full_mat.nnz, "shape:", full_mat.shape)

    # ---- Сохранение кэшей ----
    print("Saving caches...")
    bi.to_parquet(CACHE + r"\items.parquet")
    # В train оставляем только нужные колонки (запросы и кликнутые товары).
    keep = ["n_query", "s_query", "n_infm", "s_infm", "search_location_id",
            "search_category", "item_id"]
    tr[keep].to_parquet(CACHE + r"\train.parquet")
    bq.to_parquet(CACHE + r"\benchmark_queries.parquet")

    with open(CACHE + r"\token_index.pkl", "wb") as f:
        pickle.dump(token_index, f)
    sp.save_npz(CACHE + r"\full_mat.npz", full_mat.tocsr())
    sp.save_npz(CACHE + r"\title_mat.npz", title_mat.tocsr())
    # Длины документов нужны BM25 для нормализации по длине (avgdl, параметр b).
    np.save(CACHE + r"\doc_len.npy", np.asarray(full_mat.sum(axis=1)).ravel())
    np.save(CACHE + r"\titel_doc_len.npy", np.asarray(title_mat.sum(axis=1)).ravel())

    print("Done. items:", len(bi), "tokens:", len(token_index))


if __name__ == "__main__":
    main()