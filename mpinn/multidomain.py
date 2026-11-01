"""MPINN: мультидоменная PINN на Equinox в функциональном стиле.

Контейнер — только поля. Всё остальное — свободные функции.
bc_configs — аргумент fit_*/mpinn_loss_fn (не поле MPINN).

weights = [w_pde, w_dir, w_neu, w_rob, w_if] — глобальные по типу BC.
Домены используют loss_fn_split_bc; веса внутри PINN фиктивны (1, 1, 1, 1),
глобальная балансировка — на уровне mpinn.weights.
"""

from __future__ import annotations

import math
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from .pinn_core import compute_metrics
from .losses import loss_fn_split_bc
from .bc import _lambda_vals


# ============================================================ MPINN

class MPINN(eqx.Module):
    """Только поля: список PINN + геометрия интерфейсов + веса лоссов."""
    pinns: tuple
    interface_pts: tuple
    interface_n: tuple
    interface_pairs: tuple = eqx.field(static=True)

    weights: jax.Array          # [w_pde, w_dir, w_neu, w_rob, w_if]

    def __init__(self, pinns, interface_data,
                 weights=(1.0, 1.0, 1.0, 1.0, 1.0)):
        self.pinns = tuple(pinns)
        self.interface_pts = tuple(d[0] for d in interface_data)
        self.interface_n = tuple(d[1] for d in interface_data)
        self.interface_pairs = tuple((int(d[2]), int(d[3])) for d in interface_data)
        self.weights = jnp.asarray(weights)

        n = len(self.pinns)
        for i, j in self.interface_pairs:
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"interface pair ({i},{j}) вне range({n})")

    @property
    def pde_weight(self):
        return self.weights[0]

    @property
    def bc_dir_weight(self):
        return self.weights[1]

    @property
    def bc_neu_weight(self):
        return self.weights[2]

    @property
    def bc_rob_weight(self):
        return self.weights[3]

    @property
    def if_weight(self):
        return self.weights[4]

    @property
    def bc_weight(self):
        """Суммарный вес BC (для совместимости / отчётов)."""
        return self.weights[1] + self.weights[2] + self.weights[3]


# ============================================================ interface loss

def interface_loss(pinn_i, pinn_j, pts, n):
    """continuity_T + continuity_flux на одном интерфейсе.

    lam берётся из pinn_i.lam / pinn_j.lam; поддерживает скаляр,
    массив и callable(T) (через _lambda_vals).
    """
    T1 = pinn_i.net(pts).ravel()
    T2 = pinn_j.net(pts).ravel()
    cont_T = jnp.mean((T1 - T2) ** 2)

    def dT_dn(model, x, n):
        return jax.jvp(lambda y: model(y[None]).ravel()[0], (x,), (n,))[1]

    n2 = -n
    dT1n = jax.vmap(dT_dn, in_axes=(None, 0, 0))(pinn_i.net, pts, n)
    dT2n = jax.vmap(dT_dn, in_axes=(None, 0, 0))(pinn_j.net, pts, n2)

    lam_i = _lambda_vals(pinn_i.net, pts, pinn_i.lam)
    lam_j = _lambda_vals(pinn_j.net, pts, pinn_j.lam)
    cont_flux = jnp.mean((lam_i * dT1n + lam_j * dT2n) ** 2)

    return cont_T + cont_flux


# ============================================================ loss

def mpinn_loss_fn(mpinn, x_collocations, bc_configs,
                  source_fns=()):
    """Глобальный лосс MPINN.

    Требует loss_fn_split_bc во всех доменах (aux — jnp.array формы (4,)).
    bc_configs: tuple[tuple[BC, ...], ...] — BC по доменам.
    source_fns: tuple[float | callable, ...] — source_fn по доменам.

    Возвращает (total, (pde, bc_total, bc_dir, bc_neu, bc_rob, if_total)).
    """
    if not source_fns:
        source_fns = (0.0,) * len(mpinn.pinns)

    L_pde = jnp.array(0.0)
    L_dir = jnp.array(0.0)
    L_neu = jnp.array(0.0)
    L_rob = jnp.array(0.0)

    for idx, (pinn, x_d, bcs, src) in enumerate(zip(
        mpinn.pinns, x_collocations, bc_configs, source_fns,
    )):
        _, aux = loss_fn_split_bc(pinn, x_d, bcs, src)
        if not (hasattr(aux, "shape") and aux.shape == (4,)):
            raise TypeError(
                f"Домен {idx}: aux имеет форму "
                f"{getattr(aux, 'shape', type(aux).__name__)}, "
                f"ожидалось (4,) от loss_fn_split_bc. "
                f"Проверьте PINN(..., loss_fn=loss_fn_split_bc)."
            )
        L_pde = L_pde + aux[0]
        L_dir = L_dir + aux[1]
        L_neu = L_neu + aux[2]
        L_rob = L_rob + aux[3]

    L_if = jnp.array(0.0)
    for pts, n, (i, j) in zip(
        mpinn.interface_pts, mpinn.interface_n, mpinn.interface_pairs,
    ):
        L_if = L_if + interface_loss(mpinn.pinns[i], mpinn.pinns[j], pts, n)

    L_bc_total = L_dir + L_neu + L_rob

    losses_vec = jnp.array([L_pde, L_dir, L_neu, L_rob, L_if])
    total = jnp.dot(mpinn.weights, losses_vec)

    return total, (L_pde, L_bc_total, L_dir, L_neu, L_rob, L_if)


# ============================================================ оптимизация

def init_opt_states(mpinn, txs):
    if len(txs) != len(mpinn.pinns):
        raise ValueError(
            f"len(txs)={len(txs)} != len(mpinn.pinns)={len(mpinn.pinns)}"
        )
    return tuple(
        tx.init(eqx.filter(p, eqx.is_inexact_array))
        for tx, p in zip(txs, mpinn.pinns)
    )


def snapshot(mpinn):
    return jax.tree.map(jnp.copy, mpinn)


@eqx.filter_jit
def train_step(mpinn, txs, opt_states, x_collocations, bc_configs,
               source_fns=()):
    """Шаг обучения. Веса берутся из полей mpinn."""
    n = len(mpinn.pinns)
    if len(txs) != n or len(opt_states) != n:
        raise ValueError(
            f"len(txs)={len(txs)}, len(opt_states)={len(opt_states)}, "
            f"len(pinns)={n} — должны совпадать."
        )

    def closure(m):
        total, (L_pde, L_bc, L_dir, L_neu, L_rob, L_if) = mpinn_loss_fn(
            m, x_collocations, bc_configs, source_fns,
        )
        return total, (L_pde, L_bc, L_dir, L_neu, L_rob, L_if)

    (total, aux), grads = eqx.filter_value_and_grad(closure, has_aux=True)(mpinn)

    new_pinns, new_states = [], []
    for i, (tx, os) in enumerate(zip(txs, opt_states)):
        pinn = mpinn.pinns[i]
        params = eqx.filter(pinn, eqx.is_inexact_array)
        grad_i = eqx.filter(grads.pinns[i], eqx.is_inexact_array)
        updates, new_os = tx.update(grad_i, os, params)
        new_pinns.append(eqx.apply_updates(pinn, updates))
        new_states.append(new_os)

    new_mpinn = eqx.tree_at(lambda m: m.pinns, mpinn, tuple(new_pinns))
    return new_mpinn, tuple(new_states), total, aux


# ============================================================ predict / evaluate

@eqx.filter_jit
def predict(mpinn, geometries, x_test):
    """Предсказание по доменам с маскировкой."""
    x = jnp.atleast_1d(x_test)
    if x.ndim == 1:
        dim = getattr(geometries[0], "dim", None)
        if dim is None:
            raise AttributeError(
                "geometries[0] не имеет .dim; передайте x_test формы (N, D)."
            )
        x = x.reshape(-1, dim)

    t_pred = jnp.full((x.shape[0],), jnp.nan)
    for i, pinn in enumerate(mpinn.pinns):
        geom = geometries[i]
        if hasattr(geom, "is_inside"):
            mask = geom.is_inside(x)
        else:
            bmin, bmax = geom.bounds
            mask = jnp.all((x >= bmin) & (x <= bmax), axis=1)
        mask = mask & jnp.isnan(t_pred)
        t_pred = jnp.where(mask, pinn(x).ravel(), t_pred)

    return t_pred.reshape(-1, 1)


def evaluate(mpinn, geometries, x_test, exact_fn):
    """Оценка модели: (metrics, T_pred, T_exact).

    Ошибка, если тестовые точки не попали ни в один домен.
    """
    t_pred = predict(mpinn, geometries, x_test).ravel()
    n_nan = int(jnp.sum(jnp.isnan(t_pred)))
    if n_nan > 0:
        raise ValueError(
            f"{n_nan} из {len(t_pred)} тестовых точек вне всех доменов; "
            f"predict вернул nan."
        )
    t_exact = exact_fn(x_test).ravel()
    return compute_metrics(t_pred, t_exact), t_pred, t_exact


# ============================================================ fit

def fit(mpinn, txs, opt_states, x_collocations, bc_configs, epochs,
        source_fns=(), log_interval=100, print_log=False, step_offset=0):
    """Базовый цикл обучения MPINN.

    history: steps, total_loss, pde, bc_total, bc_dir, bc_neu, bc_rob, if_total.
    """
    history = {k: [] for k in (
        "steps", "total_loss", "pde", "bc_total",
        "bc_dir", "bc_neu", "bc_rob", "if_total",
    )}
    t0 = time.perf_counter()

    for step in range(epochs):
        mpinn, opt_states, total, aux = train_step(
            mpinn, txs, opt_states, x_collocations, bc_configs, source_fns,
        )
        if step % log_interval == 0 or step == epochs - 1:
            L_pde, L_bc, L_dir, L_neu, L_rob, L_if = [float(v) for v in aux]
            history["steps"].append(step_offset + step)
            history["total_loss"].append(float(total))
            history["pde"].append(L_pde)
            history["bc_total"].append(L_bc)
            history["bc_dir"].append(L_dir)
            history["bc_neu"].append(L_neu)
            history["bc_rob"].append(L_rob)
            history["if_total"].append(L_if)

            if print_log:
                print(f"step {step_offset + step:6d} | "
                      f"total={float(total):.3e}  "
                      f"pde={L_pde:.3e}  "
                      f"bc={L_bc:.3e} (dir={L_dir:.2e} neu={L_neu:.2e} rob={L_rob:.2e})  "
                      f"if={L_if:.3e}")

    return mpinn, opt_states, history, time.perf_counter() - t0


def fit_adaptive(mpinn, txs, opt_states, x_collocations, bc_configs,
                 geometries, x_test, exact_fn,
                 target_metric=1e-3, metric_key='rl2',
                 block_epochs=500, max_epochs=1_000,
                 log_interval=500, patience=20):
    """Блочное обучение с early-stop по метрике."""
    if block_epochs <= 0:
        raise ValueError(f"block_epochs must be > 0, got {block_epochs}")

    t0 = time.perf_counter()
    history = {}
    total_epochs = 0
    best_metric = float("inf")
    best_mpinn, best_states, best_epoch, no_improve = mpinn, opt_states, 0, 0

    while total_epochs < max_epochs:
        block = min(block_epochs, max_epochs - total_epochs)

        mpinn, opt_states, block_h, _ = fit(
            mpinn, txs, opt_states, x_collocations, bc_configs, block,
            log_interval=log_interval, step_offset=total_epochs,
        )
        total_epochs += block

        for k, v in block_h.items():
            history.setdefault(k, []).extend(v)

        metrics_dict, _, _ = evaluate(mpinn, geometries, x_test, exact_fn)
        current_metric = float(metrics_dict[metric_key])

        if current_metric < best_metric:
            best_metric, best_mpinn, best_states = current_metric, mpinn, opt_states
            best_epoch, no_improve = total_epochs, 0
        else:
            no_improve += 1

        print(f"[block {total_epochs:6d}] "
              f"{metric_key.upper()}={current_metric:.4e}  "
              f"pde={history['pde'][-1]:.3e}  "
              f"bc={history['bc_total'][-1]:.3e}  "
              f"if={history['if_total'][-1]:.3e}  "
              f"(best={best_metric:.4e} @ {best_epoch}, "
              f"patience={no_improve}/{patience})")

        if current_metric < target_metric:
            print(f"Target {metric_key.upper()}={target_metric:.2e} @ epoch {total_epochs}")
            break
        if no_improve >= patience:
            print(f"Early stop: no progress for {patience} blocks")
            break

    history['epochs_trained'] = total_epochs
    return best_mpinn, best_states, history, time.perf_counter() - t0


def fit_adaptive_weights(mpinn, txs, opt_states, x_collocations, bc_configs,
                         geometries, x_test, exact_fn,
                         target_metric=1e-3, metric_key='rl2',
                         block_epochs=500, max_epochs=50_000,
                         log_interval=500, patience=20,
                         smoothing=0.5, max_weight=1000.0):
    """Адаптивное обучение MPINN с loss-ratio балансировкой весов BC/IF.

    w_pde фиксирован = 1.0.
    w_dir, w_neu, w_rob, w_if ← smoothing·w + (1-smoothing)·L_*/L_pde.
    """
    if block_epochs <= 0:
        raise ValueError(f"block_epochs must be > 0, got {block_epochs}")
    if not (0.0 <= smoothing <= 1.0):
        raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")

    t0 = time.perf_counter()
    history = {}
    total_epochs = 0
    best_metric = float("inf")
    best_mpinn, best_states, best_epoch, no_improve = mpinn, opt_states, 0, 0
    lo, hi = 1.0 / max_weight, max_weight

    while total_epochs < max_epochs:
        block = min(block_epochs, max_epochs - total_epochs)

        mpinn, opt_states, block_h, _ = fit(
            mpinn, txs, opt_states, x_collocations, bc_configs, block,
            log_interval=log_interval, step_offset=total_epochs,
        )
        total_epochs += block

        for k, v in block_h.items():
            history.setdefault(k, []).extend(v)

        # --- обновление весов (PDE-anchored) ---
        l_pde = history["pde"][-1]
        if not math.isfinite(l_pde) or l_pde <= 0:
            l_pde = 1e-6
        raw = [
            history["bc_dir"][-1]   / l_pde,
            history["bc_neu"][-1]   / l_pde,
            history["bc_rob"][-1]   / l_pde,
            history["if_total"][-1] / l_pde,
        ]
        w_old = [float(v) for v in mpinn.weights[1:]]
        w_new = [
            max(lo, min(hi, smoothing * w + (1.0 - smoothing) * r))
            for w, r in zip(w_old, raw)
        ]
        new_weights = jnp.asarray(
            [1.0] + w_new, dtype=mpinn.weights.dtype,
        )
        mpinn = eqx.tree_at(lambda m: m.weights, mpinn, new_weights)

        # --- метрика / early-stop ---
        metrics_dict, _, _ = evaluate(mpinn, geometries, x_test, exact_fn)
        current_metric = float(metrics_dict[metric_key])

        if current_metric < best_metric:
            best_metric, best_mpinn, best_states = current_metric, mpinn, opt_states
            best_epoch, no_improve = total_epochs, 0
        else:
            no_improve += 1

        print(f"[block {total_epochs:6d}] "
              f"{metric_key.upper()}={current_metric:.4e}  "
              f"w={[round(float(v), 3) for v in mpinn.weights]}  "
              f"(best={best_metric:.4e}, patience={no_improve}/{patience})")

        if current_metric < target_metric:
            print(f"Target {metric_key.upper()}={target_metric:.2e} @ epoch {total_epochs}")
            break
        if no_improve >= patience:
            print(f"Early stop: no progress for {patience} blocks")
            break

    history['epochs_trained'] = total_epochs
    return best_mpinn, best_states, history, time.perf_counter() - t0