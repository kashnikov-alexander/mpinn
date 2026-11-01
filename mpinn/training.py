"""Циклы обучения PINN.

Всё, что управляет расписанием:
    - init_opt_state, snapshot — утилиты.
    - fit — базовый цикл на epochs шагов.
    - fit_adaptive — блочный, early-stop по метрике на x_test.
    - fit_adaptive_weights — блочный, loss-ratio балансировка весов BC
      (только для loss_fn_split_bc, weights длины 4: pde, dir, neu, rob).

Модель, лоссы, один шаг, оценка — в pinn_core.py и losses.py.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .pinn_core import PINN, train_step, evaluate


# ============================================================ утилиты

def init_opt_state(pinn: PINN, tx: optax.GradientTransformation) -> optax.OptState:
    params = eqx.filter(pinn, eqx.is_inexact_array)
    return tx.init(params)


def snapshot(pinn: PINN) -> PINN:
    return jax.tree.map(jnp.copy, pinn)


# ============================================================ обучение

def fit(pinn, tx, opt_state, x_collocation, bc_configs, epochs,
        source_fn=0.0, log_interval=100, print_log=False, step_offset=0):
    """Базовый цикл обучения на epochs шагов.

    Returns
    -------
    (pinn, opt_state, history, elapsed_seconds)

    history: dict[str, list]
        'steps', 'total_loss', 'pde', 'bc_total',
        далее — по одному ключу на BC (classic: 'bc_i') или
        на тип BC (split: 'bc_dir', 'bc_neu', 'bc_rob').
    """
    start_time = time.perf_counter()

    n_bc = len(bc_configs)
    loss_names = ["pde", "bc_total"] + [f"bc_{i}" for i in range(n_bc)]
    history = {"steps": [], "total_loss": []}
    for name in loss_names:
        history[name] = []

    for step in range(epochs):
        pinn, opt_state, total_loss, aux = train_step(
            pinn, tx, opt_state, x_collocation, bc_configs, source_fn
        )

        if step % log_interval == 0 or step == epochs - 1:
            if isinstance(aux, tuple):
                pde_val = float(aux[0])
                bc_vals = [float(v) for v in aux[1:]]
                bc_keys = [f"bc_{i}" for i in range(len(bc_vals))]
            else:  # split: [pde, dir, neu, rob]
                arr = [float(v) for v in aux]
                pde_val = arr[0]
                bc_vals = arr[1:]
                bc_keys = ["bc_dir", "bc_neu", "bc_rob"]

            bc_total = float(sum(bc_vals)) if bc_vals else 0.0
            total_val = float(total_loss)

            history["steps"].append(step_offset + step)
            history["total_loss"].append(total_val)
            history["pde"].append(pde_val)
            history["bc_total"].append(bc_total)
            for key, val in zip(bc_keys, bc_vals):
                history[key].append(val)

            if print_log:
                line = f"\rstep: {step_offset + step} | PDE: {pde_val:.4f} | BC: {bc_total:.4f}"
                for key, v in zip(bc_keys, bc_vals):
                    line += f" | {key}: {v:.4f}"
                print(line.ljust(120))

    return pinn, opt_state, history, time.perf_counter() - start_time


def fit_adaptive(pinn, tx, opt_state, x_collocation, bc_configs,
                 x_test, exact_fn,
                 target_metric=1e-3, block_epochs=50, max_epochs=100,
                 log_interval=100, source_fn=0.0, metric_key='rl2'):
    """Блочное обучение с early-stop по метрике на x_test."""
    MAX_PATIENCE = 10
    start_time = time.perf_counter()

    history: dict[str, list] = {}
    total_epochs = 0
    best_metric = float("inf")
    best_snapshot = snapshot(pinn)
    best_opt_state = opt_state
    patience_counter = 0

    while total_epochs < max_epochs:
        block = min(block_epochs, max_epochs - total_epochs)
        block_start = total_epochs

        pinn, opt_state, block_history, _ = fit(
            pinn, tx, opt_state, x_collocation, bc_configs, block,
            source_fn=source_fn, log_interval=log_interval,
            step_offset=block_start,
        )
        total_epochs += block

        for key, values in block_history.items():
            history.setdefault(key, []).extend(values)

        metrics_dict, _, _ = evaluate(pinn, x_test, exact_fn)
        current_metric = float(metrics_dict[metric_key])

        if current_metric < best_metric:
            best_metric = current_metric
            best_snapshot = snapshot(pinn)
            best_opt_state = opt_state
            patience_counter = 0
        else:
            patience_counter += 1

        pde_val = block_history.get("pde", [float("nan")])[-1]
        bc_val  = block_history.get("bc_total", [float("nan")])[-1]

        print(f"[block {total_epochs:6d}] "
              f"{metric_key.upper()}={current_metric:.4f}  "
              f"pde={pde_val:.4f}  bc={bc_val:.4f}  "
              f"(best={best_metric:.4f}, patience={patience_counter}/{MAX_PATIENCE})")

        if current_metric < target_metric:
            print(f"Target {metric_key.upper()}={target_metric:.2e} @ epoch {total_epochs}")
            break
        if patience_counter >= MAX_PATIENCE:
            print(f"Early stop: no progress for {MAX_PATIENCE} blocks")
            break

    history['epochs_trained'] = total_epochs
    return best_snapshot, best_opt_state, history, time.perf_counter() - start_time


def fit_adaptive_weights(pinn, tx, opt_state, x_collocation, bc_configs,
                         max_epochs=50_000, block_epochs=100,
                         x_test=None, exact_fn=None,
                         target_metric=None, metric_key='rl2', patience=20,
                         log_interval=100, source_fn=0.0,
                         smoothing=0.5, max_weight=1000.0):
    """Блочное обучение с loss-ratio балансировкой весов BC (PDE-anchored).

    Схема:   w_bc ← smoothing·w_bc + (1-smoothing) · L_bc / L_pde
    Для каждого из dir/neu/rob независимо.

    Требует:
        loss_fn = loss_fn_split_bc (weights длины 4: pde, dir, neu, rob);
        history с ключами 'pde', 'bc_dir', 'bc_neu', 'bc_rob'.

    weights[0] (PDE) фиксируется = 1.0.
    Веса в history не пишутся — только в print.
    x_test/exact_fn → early-stop по метрике.
    """
    n_w = int(len(pinn.weights))
    if n_w != 4:
        raise ValueError(
            f"fit_adaptive_weights ожидает weights длины 4 "
            f"(pde, dir, neu, rob), получено {n_w}. "
            f"Для другого лосса напишите свою fit_adaptive_weights."
        )
    if not (0.0 <= smoothing <= 1.0):
        raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")

    use_early_stop = (x_test is not None and exact_fn is not None)
    start_time = time.perf_counter()

    history: dict[str, list] = {}
    total_epochs = 0
    best_metric = float("inf")
    best_snapshot = snapshot(pinn)
    best_opt_state = opt_state
    patience_counter = 0

    lo, hi = 1.0 / max_weight, max_weight

    while total_epochs < max_epochs:
        block = min(block_epochs, max_epochs - total_epochs)
        block_start = total_epochs

        pinn, opt_state, block_history, _ = fit(
            pinn, tx, opt_state, x_collocation, bc_configs, block,
            source_fn=source_fn, log_interval=log_interval,
            step_offset=block_start,
        )
        total_epochs += block

        for key, values in block_history.items():
            history.setdefault(key, []).extend(values)

        # --- обновление весов BC (PDE-anchored) ---
        l_pde = float(block_history["pde"][-1])
        if not jnp.isfinite(l_pde) or l_pde <= 0:
            l_pde = 1e-6
        raw = [
            float(block_history[k][-1]) / l_pde
            for k in ("bc_dir", "bc_neu", "bc_rob")
        ]
        w_old = [float(v) for v in pinn.weights[1:]]
        w_bc = [
            max(lo, min(hi, smoothing * w + (1.0 - smoothing) * r))
            for w, r in zip(w_old, raw)
        ]
        new_weights = jnp.asarray(
            [1.0] + w_bc, dtype=pinn.weights.dtype,
        )
        pinn = eqx.tree_at(lambda p: p.weights, pinn, new_weights)

        # --- early-stop / логирование ---
        if use_early_stop:
            metrics_dict, _, _ = evaluate(pinn, x_test, exact_fn)
            current_metric = float(metrics_dict[metric_key])

            if current_metric < best_metric:
                best_metric = current_metric
                best_snapshot = snapshot(pinn)
                best_opt_state = opt_state
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"[block {total_epochs:6d}] "
                  f"{metric_key.upper()}={current_metric:.4e}  "
                  f"w={[round(float(v), 3) for v in pinn.weights]}  "
                  f"(best={best_metric:.4e}, patience={patience_counter}/{patience})")

            if target_metric is not None and current_metric < target_metric:
                print(f"Target {metric_key.upper()}={target_metric:.2e} @ epoch {total_epochs}")
                break
            if patience_counter >= patience:
                print(f"Early stop: no progress for {patience} blocks")
                break
        else:
            print(f"[block {total_epochs:6d}] "
                  f"w={[round(float(v), 3) for v in pinn.weights]}")

    history['epochs_trained'] = total_epochs
    return best_snapshot, best_opt_state, history, time.perf_counter() - start_time