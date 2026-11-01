"""Модель PINN на JAX + Equinox, один шаг обучения, оценка.

Сюда входит:
    - FCNet, ScaledNet — сети.
    - PINN — контейнер (net, lam, weights, pde_fn, loss_fn).
    - train_step — один шаг оптимизатора.
    - compute_metrics, evaluate — оценка качества.

Лоссы — в losses.py. Циклы обучения — в training.py.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal


# ============================================================ сети

class FCNet(eqx.Module):
    """Полносвязная сеть. (din,) -> (dout,) или (N, din) -> (N, dout)."""
    layers: list
    linear_out: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        num_layers: int,
        activation: Callable,
        key: jax.Array,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
    ):
        if weight_init is None:
            weight_init = glorot_normal(in_axis=-1, out_axis=-2)
        if bias_init is None:
            bias_init = jax.nn.initializers.zeros

        keys = jax.random.split(key, num_layers + 1)
        self.layers = [
            eqx.nn.Linear(
                din if i == 0 else dmid, dmid, key=keys[i],
                weight_init=weight_init, bias_init=bias_init,
            )
            for i in range(num_layers)
        ]
        self.linear_out = eqx.nn.Linear(
            dmid, dout, key=keys[-1],
            weight_init=weight_init, bias_init=bias_init,
        )
        self.activation = activation

    def _forward_one(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.linear_out(x)

    def __call__(self, x):
        if x.ndim == 1:
            return self._forward_one(x)
        return jax.vmap(self._forward_one)(x)


class ScaledNet(eqx.Module):
    """Нормирует вход (координаты) в [0, 1] и выход θ -> T."""
    base: eqx.Module
    x_min: jax.Array
    x_max: jax.Array
    T_ref:   float = eqx.field(static=True)
    T_scale: float = eqx.field(static=True)

    def __init__(self, base, x_min, x_max, T_ref=0.0, T_scale=1.0):
        self.base = base
        self.x_min = jnp.asarray(x_min, dtype=jnp.float32)
        self.x_max = jnp.asarray(x_max, dtype=jnp.float32)
        self.T_ref   = float(T_ref)
        self.T_scale = float(T_scale)

    def __call__(self, x):
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        theta = self.base(x_norm)
        return self.T_ref + self.T_scale * theta


def dimensionless(scale):
    """Декоратор: умножает результат функции на масштабный коэффициент."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs) * scale
        return wrapper
    return decorator


# ============================================================ PINN

class PINN(eqx.Module):
    """Контейнер модели.

    loss_fn: callable(pinn, x_colloc, bc_configs, source_fn) -> (total, aux)
        Контракт по len(weights) задаёт сам loss_fn (см. losses.py).
    weights: jax.Array любой длины; семантику задаёт loss_fn.
    """
    net: eqx.Module

    lam_val: jax.Array
    lam_fn: Optional[Callable] = eqx.field(static=True)

    weights: jax.Array

    pde_fn: Callable = eqx.field(static=True)
    loss_fn: Callable = eqx.field(static=True)

    def __init__(self, net, pde_fn, loss_fn, lam=1.0, weights=(1.0, 1.0)):
        self.net = net
        self.pde_fn = pde_fn
        self.loss_fn = loss_fn

        if callable(lam):
            self.lam_val = jnp.array(1.0)
            self.lam_fn = lam
        else:
            self.lam_val = jnp.asarray(lam)
            self.lam_fn = None

        self.weights = jnp.asarray(weights)

    @property
    def lam(self):
        if self.lam_fn is not None:
            return self.lam_fn
        return self.lam_val

    @property
    def pde_weight(self):
        return self.weights[0]

    @property
    def bc_weight(self):
        return self.weights[1]

    def __call__(self, x):
        return self.net(x)


# ============================================================ один шаг

@eqx.filter_jit
def train_step(pinn: PINN, tx, opt_state, x_collocation, bc_configs, source_fn=0.0):
    """Один шаг обучения. bc_configs — статический tuple, не поле PINN."""
    def closure(p):
        return p.loss_fn(p, x_collocation, bc_configs, source_fn)

    (total, aux), grads = eqx.filter_value_and_grad(closure, has_aux=True)(pinn)

    params = eqx.filter(pinn, eqx.is_inexact_array)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_pinn = eqx.apply_updates(pinn, updates)
    return new_pinn, new_opt_state, total, aux


# ============================================================ оценка

def compute_metrics(T_pred, T_exact):
    """Метрики от двух массивов. Все значения — Python float."""
    diff = T_pred - T_exact
    mse_jnp = jnp.mean(diff ** 2)
    mse = float(mse_jnp)
    rmse = float(jnp.sqrt(mse_jnp))
    mae = float(jnp.mean(jnp.abs(diff)))
    max_error = float(jnp.max(jnp.abs(diff)))
    mape = float(jnp.mean(jnp.abs(diff / (jnp.abs(T_exact) + 1e-8))))
    rl2 = float(jnp.linalg.norm(diff) / (jnp.linalg.norm(T_exact) + 1e-12))
    return {
        "rl2":       rl2,
        "mape":      mape,
        "max_error": max_error,
        "mae":       mae,
        "mse":       mse,
        "rmse":      rmse,
    }


def evaluate(pinn, x_test, exact_fn):
    """Возвращает (metrics, T_pred, T_exact). metrics: dict[str, float]."""
    T_pred = pinn(x_test).ravel()
    T_exact = exact_fn(x_test).ravel()
    metrics = compute_metrics(T_pred, T_exact)
    return metrics, T_pred, T_exact