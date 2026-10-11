"""
Ядро реализации PINN на основе JAX и Equinox.
Поддерживает 1D, 2D и 3D задачи.

Архитектура в PyTorch-стиле:
    - PINN — тонкий контейнер (поля + __call__).
    - Обучение, оценка, метрики — свободные функции.
    - Оптимизатор (tx + opt_state) — снаружи модели.
    - lam / pde_weight / bc_weight — обычные поля-массивы (не static),
      поэтому работают и как фиксированные константы, и как обучаемые
      параметры (inverse problem, adaptive weights) без изменения API.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


# ============================================================ сети

class FCNet(eqx.Module):
    """Полносвязная сеть. Принимает (din,) или (N, din), возвращает (dout,) или (N, dout)."""

    layers: list
    linear_out: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)

    def __init__(self, din, dmid, dout, num_layers, activation, key):
        keys = jax.random.split(key, num_layers + 1)
        self.layers = [
            eqx.nn.Linear(din if i == 0 else dmid, dmid, key=keys[i])
            for i in range(num_layers)
        ]
        self.linear_out = eqx.nn.Linear(dmid, dout, key=keys[-1])
        self.activation = activation

    def _forward_one(self, x):
        """Одна точка: x = (din,) -> (dout,)."""
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.linear_out(x)

    def __call__(self, x):
        """(din,) -> (dout,)   или   (N, din) -> (N, dout)."""
        if x.ndim == 1:
            return self._forward_one(x)
        return jax.vmap(self._forward_one)(x)

# ============================================================ прямоугольник

class ScaledNet(eqx.Module):
    base: eqx.Module
    x_min: jax.Array
    x_max: jax.Array

    def __init__(self, base, x_min, x_max):
        self.base = base
        self.x_min = jnp.asarray(x_min, dtype=jnp.float32)
        self.x_max = jnp.asarray(x_max, dtype=jnp.float32)

    def __call__(self, x):
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        return self.base(x_norm)


# ============================================================ кольцо

class RadialScaledNet(eqx.Module):
    """
    Кольцевая область: нормирует только вход. Выход сети — физическая T.

    Вход:
        x = (r, theta)          — 2D annulus
        x = (r, theta, z)       — 3D annulus (z опционально)

    Нормировка:
        r      -> r_norm  (min-max или log)
        theta  -> (cos theta, sin theta)
        z      -> z_norm  (min-max, если задан)

    din для base:
        2D: 3   (r_norm, cos, sin)
        3D: 4   (r_norm, cos, sin, z_norm)

    R_inner сохраняется как аддитивный сдвиг (или база логарифма),
    поэтому r = R_inner != 0 корректно попадает в pde_fn как 1/r.
    """

    base: eqx.Module
    R_inner: float = eqx.field(static=True)
    R_outer: float = eqx.field(static=True)
    log_radius: bool = eqx.field(static=True)
    has_z: bool = eqx.field(static=True)
    z_min: float = eqx.field(static=True)
    z_max: float = eqx.field(static=True)

    def __init__(self, base, R_inner, R_outer,
                 log_radius=False, z_min=None, z_max=None):
        self.base = base
        self.R_inner = float(R_inner)
        self.R_outer = float(R_outer)
        self.log_radius = bool(log_radius)

        self.has_z = (z_min is not None) and (z_max is not None)
        self.z_min = float(z_min) if z_min is not None else 0.0
        self.z_max = float(z_max) if z_max is not None else 1.0

    def __call__(self, x):
        dtype = x.dtype

        r = x[..., 0:1]
        theta = x[..., 1:2]

        if self.log_radius:
            r_norm = jnp.log(r / jnp.asarray(self.R_inner, dtype)) / \
                     jnp.log(jnp.asarray(self.R_outer / self.R_inner, dtype))
        else:
            r_norm = (r - jnp.asarray(self.R_inner, dtype)) / \
                     jnp.asarray(self.R_outer - self.R_inner, dtype)

        parts = [r_norm, jnp.cos(theta), jnp.sin(theta)]

        if self.has_z:
            z = x[..., 2:3]
            z_norm = (z - jnp.asarray(self.z_min, dtype)) / \
                     jnp.asarray(self.z_max - self.z_min, dtype)
            parts.append(z_norm)

        x_norm = jnp.concatenate(parts, axis=-1)
        return self.base(x_norm)


# ============================================================ утилиты

from typing import Callable, Optional


class BCConfig(eqx.Module):
    """
    Одно граничное условие.
      fn, params  — static (callable + tuple из hashable-скаляров)
      points, normals — данные (JAX-массивы, листья pytree)
    """
    fn: Callable = eqx.field(static=True)
    params: tuple = eqx.field(static=True)
    points: jax.Array
    normals: Optional[jax.Array] = None


def _freeze_bc_configs(bc_configs: list[dict]) -> tuple:
    return tuple(
        BCConfig(
            fn=c["fn"],
            params=tuple(sorted(c.get("params", {}).items())),
            points=jnp.asarray(c["points"]),
            normals=jnp.asarray(c["normals"]) if c.get("normals") is not None else None,
        )
        for c in bc_configs
    )


class PINN(eqx.Module):
    net: eqx.Module
    lam: jax.Array
    pde_weight: jax.Array
    bc_weight: jax.Array
    pde_fn: Callable = eqx.field(static=True)
    bc_configs: tuple = ()          # НЕ static — теперь pytree

    def __init__(self, net, pde_fn, bc_configs,
                 lam=1.0, pde_weight=1.0, bc_weight=1.0):
        self.net = net
        self.pde_fn = pde_fn
        self.bc_configs = _freeze_bc_configs(bc_configs)
        self.lam = jnp.asarray(lam)
        self.pde_weight = jnp.asarray(pde_weight)
        self.bc_weight = jnp.asarray(bc_weight)

    def __call__(self, x):
        return self.net(x)





# ============================================================ оптимизация

def init_opt_state(pinn: PINN, tx: optax.GradientTransformation) -> optax.OptState:
    """Инициализировать состояние оптимизатора для обучаемых листьев PINN."""
    params = eqx.filter(pinn, eqx.is_inexact_array)
    return tx.init(params)


def snapshot(pinn: PINN) -> PINN:
    """Копия модели (для best-checkpoint в fit_adaptive / fit_loss)."""
    return jax.tree.map(jnp.copy, pinn)


def restore(pinn: PINN, saved: PINN) -> PINN:
    """
    Восстановить модель из снапшота.
    Снапшот хранит весь PyTree целиком (static-поля копируются по ссылке),
    поэтому достаточно вернуть его.
    """
    return saved


# ============================================================ лоссы

def loss_fn(pinn: PINN, x_collocation, source_fn=0.0):
    """
    Чистая функция лосса. Возвращает (total, (loss_pde, *loss_bcs)).

    Веса pde_weight / bc_weight берутся из полей pinn и применяются
    только к total — компоненты в aux остаются «сырыми», чтобы history
    был сопоставим между запусками с разными весами.
    """
    residuals_pde = pinn.pde_fn(pinn.net, x_collocation, pinn.lam, source_fn)
    loss_pde = jnp.mean(residuals_pde ** 2)

    loss_bcs = []
    for bc in pinn.bc_configs:
        params = dict(bc.params)
        if bc.normals is not None:
            residuals = bc.fn(pinn.net, bc.points, bc.normals, **params)
        else:
            residuals = bc.fn(pinn.net, bc.points, **params)
        loss_bcs.append(jnp.mean(residuals ** 2))

    loss_bc_total = sum(loss_bcs) if loss_bcs else jnp.array(0.0)
    total = pinn.pde_weight * loss_pde + pinn.bc_weight * loss_bc_total
    return total, (loss_pde, *loss_bcs)


@eqx.filter_jit
def unweighted_loss(pinn: PINN, x_collocation, source_fn=0.0):
    """Сырая сумма PDE + BC (без весов) — для early-stopping в fit_loss."""
    _, (pde, *bcs) = loss_fn(pinn, x_collocation, source_fn)
    bc_total = sum(bcs) if bcs else jnp.array(0.0)
    return pde + bc_total


# ============================================================ один шаг

@eqx.filter_jit
def train_step(pinn: PINN, tx, opt_state, x_collocation, source_fn=0.0):
    """
    Один шаг обучения. Возвращает (new_pinn, new_opt_state, total, aux).
    Обновляются все inexact-array-листья: веса сети + lam + веса лосса.
    """
    def closure(p):
        return loss_fn(p, x_collocation, source_fn)

    (total, aux), grads = eqx.filter_value_and_grad(closure, has_aux=True)(pinn)

    params = eqx.filter(pinn, eqx.is_inexact_array)
    updates, new_opt_state = tx.update(grads, opt_state, params)

    new_pinn = eqx.apply_updates(pinn, updates)
    return new_pinn, new_opt_state, total, aux


# ============================================================ обучение

def fit(pinn, tx, opt_state, x_collocation, epochs,
        source_fn=0.0, log_interval=100, print_log=False):
    """
    Базовый цикл обучения на epochs шагов.

    Returns
    -------
    (pinn, opt_state, history, elapsed_seconds)
    """
    start_time = time.perf_counter()

    n_bc = len(pinn.bc_configs)
    loss_names = ["pde", "bc_total"] + [f"bc_{i}" for i in range(n_bc)]
    history = {"total_loss": []}
    for name in loss_names:
        history[name] = []

    for step in range(epochs):
        pinn, opt_state, total_loss, aux_losses = train_step(
            pinn, tx, opt_state, x_collocation, source_fn
        )

        if step % log_interval == 0 or step == epochs - 1:
            pde_val = float(aux_losses[0])
            bc_vals = [float(v) for v in aux_losses[1:]]
            bc_total = float(sum(bc_vals)) if bc_vals else 0.0
            total_val = float(total_loss)

            history["total_loss"].append(total_val)
            history["pde"].append(pde_val)
            history["bc_total"].append(bc_total)
            for i, val in enumerate(bc_vals):
                history[f"bc_{i}"].append(val)

            if print_log:
                line = (
                    f"\rstep: {step} | "
                    f"PDE: {pde_val:.3e} | "
                    f"BC: {bc_total:.3e}"
                )
                for i, v in enumerate(bc_vals):
                    line += f" | BC_{i}: {v:.3e}"
                print(line.ljust(120))

    return pinn, opt_state, history, time.perf_counter() - start_time


def fit_adaptive(pinn, tx, opt_state, x_collocation, x_test, exact_fn,
                 target_rl2=1e-3, block_epochs=50, max_epochs=5000,
                 log_interval=100, source_fn=0.0):
    """
    Блочное обучение с early-stop по относительной L2-ошибке.
    Возвращает лучшую по RL2 модель.

    Returns
    -------
    (pinn, opt_state, history, elapsed_seconds)
    """
    start_time = time.perf_counter()

    history: dict[str, list] = {}
    total_epochs = 0
    best_rl2 = float("inf")
    best_snapshot = snapshot(pinn)
    best_opt_state = opt_state

    while total_epochs < max_epochs:
        block = min(block_epochs, max_epochs - total_epochs)
        pinn, opt_state, block_history, _ = fit(
            pinn, tx, opt_state, x_collocation, block,
            source_fn=source_fn, log_interval=log_interval,
        )
        total_epochs += block

        for key, values in block_history.items():
            history.setdefault(key, []).extend(values)

        rl2 = float(rl2_error(pinn, x_test, exact_fn))
        if rl2 < best_rl2:
            best_rl2 = rl2
            best_snapshot = snapshot(pinn)
            best_opt_state = opt_state

        if rl2 < target_rl2:
            break

    pinn = restore(pinn, best_snapshot)
    return pinn, best_opt_state, history, time.perf_counter() - start_time


def fit_loss(pinn, tx, opt_state, x_collocation, epochs,
             log_interval=100, checkpoint_every=50, source_fn=0.0):
    """
    Блочное обучение с early-stop по сырому лоссу (PDE + BC, без весов).
    Возвращает лучшую по лоссу модель.

    Returns
    -------
    (pinn, opt_state, history, elapsed_seconds)
    """
    start_time = time.perf_counter()

    history: dict[str, list] = {}
    total_epochs = 0
    best_loss = float("inf")
    best_snapshot = snapshot(pinn)
    best_opt_state = opt_state

    while total_epochs < epochs:
        block = min(checkpoint_every, epochs - total_epochs)
        pinn, opt_state, block_history, _ = fit(
            pinn, tx, opt_state, x_collocation, block,
            source_fn=source_fn, log_interval=log_interval,
        )
        total_epochs += block

        for key, values in block_history.items():
            history.setdefault(key, []).extend(values)

        loss = float(unweighted_loss(pinn, x_collocation, source_fn))
        if loss < best_loss:
            best_loss = loss
            best_snapshot = snapshot(pinn)
            best_opt_state = opt_state

    pinn = restore(pinn, best_snapshot)
    return pinn, best_opt_state, history, time.perf_counter() - start_time


# ============================================================ оценка

def compute_metrics(T_pred, T_exact):
    """Чистая функция метрик от двух массивов."""
    diff = T_pred - T_exact
    mse = float(jnp.mean(diff ** 2))
    mae = float(jnp.mean(jnp.abs(diff)))
    rmse = float(jnp.sqrt(mse))
    max_error = float(jnp.max(jnp.abs(diff)))
    mape = float(jnp.mean(jnp.abs(diff / (jnp.abs(T_exact) + 1e-8))))
    rl2 = float(jnp.linalg.norm(diff) / (jnp.linalg.norm(T_exact) + 1e-12))
    return {
        "rl2":       f"{rl2:.4e}",
        "mape":      f"{mape:.4e}",
        "max_error": f"{max_error:.4e}",
        "mae":       f"{mae:.4e}",
        "mse":       f"{mse:.4e}",
        "rmse":      f"{rmse:.4e}",
    }


@eqx.filter_jit
def rl2_error(pinn, x_test, exact_fn):
    """Относительная L2-ошибка ‖T_pred − T_exact‖ / ‖T_exact‖."""
    T_pred = pinn(x_test).ravel()
    T_exact = exact_fn(x_test).ravel()
    return jnp.linalg.norm(T_pred - T_exact) / (
        jnp.linalg.norm(T_exact) + 1e-12
    )


def evaluate(pinn, x_test, exact_fn, bc_info=None):
    """
    Оценка модели. Возвращает (metrics, T_pred, T_exact).
    bc_info — необязательный dict {name: bc_type} для добавления в метрики.
    """
    T_pred = pinn(x_test).ravel()
    T_exact = exact_fn(x_test).ravel()
    metrics = compute_metrics(T_pred, T_exact)
    if bc_info is not None:
        for name, bc_type in bc_info.items():
            metrics[f"bc_{name}"] = bc_type
    return metrics, T_pred, T_exact