import os

HERE = os.path.dirname(os.path.abspath(__file__))                       # папка avito_solution
CACHE = os.path.join(HERE, "cache")                                     # кэши (матрицы, эмбеддинги)
OUT = os.path.join(HERE, "out")                                         # результаты (answer.csv и пр.)
BASE = os.path.join(os.path.dirname(HERE), "dataset_extract")           # исходные parquet-файлы