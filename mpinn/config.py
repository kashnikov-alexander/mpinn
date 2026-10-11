"""Configuration for experiments: physics parameters, model and training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import optax
from flax import nnx
import os
import pandas as pd


@dataclass
class TrainConfig:
    # Обязательные поля
    hidden_features: int
    num_layers: int
    activation_name: str
    opt_name: str
    lr: float
    num_points: int
    n_test_points: int

    # Необязательные поля
    log_interval: int = 100
    weights: tuple[float, float] = (1.0, 1.0)


def get_activation(name: str):
    """Фабрика функций активации."""
    mapping = {
        "tanh": nnx.tanh,
        "Swish": nnx.silu,
        "sin": jnp.sin,
        "GELU": nnx.gelu,
        "ReLU": nnx.relu,
        "Sigmoid": nnx.sigmoid,
    }
    return mapping.get(name, nnx.relu)


def get_optimizer(name: str, lr: float, schedule=None):
    """Фабрика оптимизаторов с поддержкой learning rate scheduler."""
    name = name.lower()

    def _wrap(base_lr):
        return schedule if schedule is not None else base_lr

    if name == "adam":
        return optax.adam(_wrap(lr))
    if name == "sgd":
        return optax.sgd(_wrap(lr))
    if name == "adagrad":
        return optax.adagrad(_wrap(lr))
    if name == "rmsprop":
        return optax.rmsprop(_wrap(lr))
    raise ValueError(f"Неизвестный оптимизатор: {name}.")


def normalize_coords(coords, x_min, x_max):
    rng = x_max - x_min + (x_max == x_min) * 1e-8
    return (coords - x_min) / rng


def normalize_temp(temps, t_max=None):
    if t_max is None:
        t_max = temps.max()
    return temps / (t_max + (t_max == 0) * 1e-8)


def denormalize_coords(coords, x_min, x_max):
    return coords * (x_max - x_min) + x_min


def denormalize_temp(temp, t_max):
    return temp * t_max


def save_results_to_csv(results, csv_path, append=True):
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

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if append and os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)

    print(f"Сохранено: {csv_path} ({len(df)} строк)")
    return df


def make_exact_fn_from_points(values):
    """
    Строит exact_fn, возвращающий фиксированный массив значений.

    Parameters
    ----------
    values : array (M,) или (M, 1)
        Значения решения в точках x_data.

    Returns
    -------
    exact_fn : callable
        Функция от (N, D), игнорирующая аргумент, возвращающая (M, 1).
    """
    values = jnp.asarray(values).reshape(-1, 1)

    def exact_fn(_):
        return values

    return exact_fn