"""Утилиты: сохранение результатов экспериментов."""

from __future__ import annotations

import os

import pandas as pd


def save_results_to_csv(results, csv_path, append=True):
    """Сохранить результаты эксперимента в CSV.

    results: dict | list[dict] | pd.DataFrame
        Одна строка (dict), список строк (list[dict]) или готовый DataFrame.
    csv_path: str
        Путь к файлу. Директория создаётся автоматически.
    append: bool
        True — дописать в конец (без заголовка).
        False — перезаписать (с заголовком).

    Returns
    -------
    pd.DataFrame — то, что было записано.
    """
    if isinstance(results, pd.DataFrame):
        df = results.copy()
    elif isinstance(results, dict):
        df = pd.DataFrame([results])
    elif isinstance(results, list):
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
    else:
        raise TypeError(f"Неподдерживаемый тип: {type(results)}")

    dirname = os.path.dirname(csv_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if append and os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)

    print(f"Сохранено: {csv_path} ({len(df)} строк)")
    return df