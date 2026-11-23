"""MPINN: мультидоменная PINN на Equinox в функциональном стиле.

Контейнер — только поля. Всё остальное — свободные функции.
bc_configs — аргумент fit_*/mpinn_loss_fn (не поле MPINN).

weights = [w_pde, w_dir, w_neu, w_rob, w_if_T, w_if_flux] — глобальные.
Домены используют loss_fn_split_bc; веса внутри PINN фиктивны (1, 1, 1, 1),
глобальная балансировка — на уровне mpinn.weights.
"""

from __future__ import annotations

import math
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from .bc import _lambda_vals
from .losses import loss_fn_split_bc
from .pinn_core import compute_metrics

# ============================================================ MPINN


class MPINN(eqx.Module):
    """Только поля: список PINN + геометрия интерфейсов + веса лоссов."""

    pinns: tuple
    interface_pts: tuple
    interface_n: tuple
    interface_pairs: tuple = eqx.field(static=True)

    weights: jax.Array  # [w_pde, w_dir, w_neu, w_rob, w_if_T, w_if_flux]

    def __init__(self, pinns, interface_data, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
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
    def if_T_weight(self):
        return self.weights[4]

    @property
    def if_flux_weight(self):
        return self.weights[5]

    @property
    def if_weight(self):
        """Суммарный вес интерфейса (для совместимости / отчётов)."""
        return self.weights[4] + self.weights[5]

    @property
    def bc_weight(self):
        """Суммарный вес BC (для совместимости / отчётов)."""
        return self.weights[1] + self.weights[2] + self.weights[3]


# ============================================================ interface loss


def interface_loss(pinn_i, pinn_j, pts, n):
    """Возвращает (continuity_T, continuity_flux) на одном интерфейсе.

    lam берётся из pinn_i.lam / pinn_j.lam.
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

    return cont_T, cont_flux


# ============================================================ loss


def mpinn_loss_fn(mpinn, x_collocations, bc_configs, source_fns=()):
    """Глобальный лосс MPINN.

    Возвращает (total, (pde, bc_total, bc_dir, bc_neu, bc_rob, if_T, if_flux)).
    """
    if not source_fns:
        source_fns = (0.0,) * len(mpinn.pinns)

    L_pde = jnp.array(0.0)
    L_dir = jnp.array(0.0)
    L_neu = jnp.array(0.0)
    L_rob = jnp.array(0.0)

    for idx, (pinn, x_d, bcs, src) in enumerate(
        zip(
            mpinn.pinns,
            x_collocations,
            bc_configs,
            source_fns,
        )
    ):
        _, aux = loss_fn_split_bc(pinn, x_d, bcs, src)
        if not (isinstance(aux, jax.Array) and aux.shape == (4,)):
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

    # среднее по доменам
    n_dom = len(mpinn.pinns)
    L_pde = L_pde / n_dom
    L_dir = L_dir / n_dom
    L_neu = L_neu / n_dom
    L_rob = L_rob / n_dom
    L_bc_total = L_dir + L_neu + L_rob

    # среднее по интерфейсам
    L_if_T = jnp.array(0.0)
    L_if_flux = jnp.array(0.0)
    n_if = len(mpinn.interface_pairs)

    if n_if > 0:
        for pts, n, (i, j) in zip(
            mpinn.interface_pts,
            mpinn.interface_n,
            mpinn.interface_pairs,
        ):
            cont_T, cont_flux = interface_loss(mpinn.pinns[i], mpinn.pinns[j], pts, n)
            L_if_T = L_if_T + cont_T
            L_if_flux = L_if_flux + cont_flux

        L_if_T = L_if_T / n_if
        L_if_flux = L_if_flux / n_if

    # 6 компонентов потерь
    losses_vec = jnp.array([L_pde, L_dir, L_neu, L_rob, L_if_T, L_if_flux])
    total = jnp.dot(mpinn.weights, losses_vec)

    # 7 элементов в aux (для логов)
    return total, (L_pde, L_bc_total, L_dir, L_neu, L_rob, L_if_T, L_if_flux)


# ============================================================ оптимизация


def init_opt_states(mpinn, txs):
    if len(txs) != len(mpinn.pinns):
        raise ValueError(f"len(txs)={len(txs)} != len(mpinn.pinns)={len(mpinn.pinns)}")
    return tuple(
        tx.init(eqx.filter(p, eqx.is_inexact_array)) for tx, p in zip(txs, mpinn.pinns)
    )


def snapshot(mpinn):
    return jax.tree.map(jnp.copy, mpinn)


@eqx.filter_jit
def train_step(mpinn, txs, opt_states, x_collocations, bc_configs, source_fns=()):
    """Шаг обучения."""
    n = len(mpinn.pinns)
    if len(txs) != n or len(opt_states) != n:
        raise ValueError(
            f"len(txs)={len(txs)}, len(opt_states)={len(opt_states)}, "
            f"len(pinns)={n} — должны совпадать."
        )

    def closure(m):
        total, (L_pde, L_bc, L_dir, L_neu, L_rob, L_if_T, L_if_flux) = mpinn_loss_fn(
            m,
            x_collocations,
            bc_configs,
            source_fns,
        )
        return total, (L_pde, L_bc, L_dir, L_neu, L_rob, L_if_T, L_if_flux)

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
    """Оценка модели: (metrics, T_pred, T_exact)."""
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


def fit(
    mpinn,
    txs,
    opt_states,
    x_collocations,
    bc_configs,
    epochs,
    source_fns=(),
    log_interval=100,
    print_log=False,
    step_offset=0,
):
    """Базовый цикл обучения MPINN."""
    t0 = time.perf_counter()

    history = {"steps": [], "total_loss": [], "loss_components": []}

    for step in range(epochs):
        mpinn, opt_states, total, aux = train_step(
            mpinn,
            txs,
            opt_states,
            x_collocations,
            bc_configs,
            source_fns,
        )

        if step % log_interval == 0 or step == epochs - 1:
            aux_tuple = tuple(float(v) for v in aux)

            history["steps"].append(step_offset + step)
            history["total_loss"].append(float(total))
            history["loss_components"].append(aux_tuple)

            if print_log:
                L_pde, L_bc, L_dir, L_neu, L_rob, L_if_T, L_if_flux = aux_tuple
                print(
                    f"step {step_offset + step:6d} | "
                    f"total={float(total):.3e}  "
                    f"pde={L_pde:.3e}  "
                    f"bc={L_bc:.3e} (dir={L_dir:.2e} neu={L_neu:.2e} rob={L_rob:.2e})  "
                    f"if_T={L_if_T:.3e}  if_flux={L_if_flux:.3e}"
                )

    return mpinn, opt_states, history, time.perf_counter() - t0


def fit_adaptive(
    mpinn,
    txs,
    opt_states,
    x_collocations,
    bc_configs,
    geometries,
    x_test,
    exact_fn,
    target_metric=1e-3,
    metric_key="rl2",
    block_epochs=500,
    max_epochs=1_000,
    log_interval=1,
    print_every=100,
    patience=100,
):
    """Блочное обучение с early-stop по метрике."""
    if block_epochs <= 0:
        raise ValueError(f"block_epochs must be > 0, got {block_epochs}")

    t0 = time.perf_counter()
    history = {"steps": [], "total_loss": [], "loss_components": []}

    total_epochs = 0
    best_metric = float("inf")
    best_mpinn, best_states, best_epoch, no_improve = mpinn, opt_states, 0, 0

    while total_epochs < max_epochs:
        block = min(block_epochs, max_epochs - total_epochs)

        mpinn, opt_states, block_h, _ = fit(
            mpinn,
            txs,
            opt_states,
            x_collocations,
            bc_configs,
            block,
            log_interval=log_interval,
            step_offset=total_epochs,
        )
        total_epochs += block

        for k, v in block_h.items():
            if k in history:
                history[k].extend(v)
            else:
                history[k] = v

        metrics_dict, _, _ = evaluate(mpinn, geometries, x_test, exact_fn)
        current_metric = float(metrics_dict[metric_key])

        if current_metric < best_metric:
            best_metric, best_mpinn, best_states = current_metric, mpinn, opt_states
            best_epoch, no_improve = total_epochs, 0
        else:
            no_improve += 1

        if total_epochs % print_every == 0 or current_metric < target_metric:
            last_aux = block_h["loss_components"][-1]
            L_pde, L_bc_total, L_dir, L_neu, L_rob, L_if_T, L_if_flux = last_aux

            print(
                f"[epoch {total_epochs:6d}] "
                f"{metric_key.upper()}={current_metric:.4e}  "
                f"pde={L_pde:.3e}  "
                f"dir={L_dir:.3e}  "
                f"neu={L_neu:.3e}  "
                f"rob={L_rob:.3e}  "
                f"if_T={L_if_T:.3e}  "
                f"if_flux={L_if_flux:.3e}  "
                f"(best={best_metric:.4e} {best_epoch}, "
                f"patience={no_improve}/{patience})"
            )

        if current_metric < target_metric:
            print(
                f"Target {metric_key.upper()}={target_metric:.2e} @ epoch {total_epochs}"
            )
            break
        if no_improve >= patience:
            print(f"Early stop: no progress for {patience} blocks")
            break

    history["epochs_trained"] = total_epochs
    return best_mpinn, best_states, history, time.perf_counter() - t0


def fit_adaptive_weights(
    mpinn,
    txs,
    opt_states,
    x_collocations,
    bc_configs,
    geometries,
    x_test,
    exact_fn,
    target_metric=1e-3,
    metric_key="rl2",
    block_epochs=500,
    max_epochs=20_000,
    log_interval=500,
    patience=10,
    smoothing=0.5,
    max_weight=1000.0,
):
    """Адаптивное обучение MPINN с балансировкой 6 весов."""
    if block_epochs <= 0:
        raise ValueError(f"block_epochs must be > 0, got {block_epochs}")
    if not (0.0 <= smoothing <= 1.0):
        raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")

    t0 = time.perf_counter()
    history = {"steps": [], "total_loss": [], "loss_components": []}

    total_epochs = 0
    best_metric = float("inf")
    best_mpinn, best_states, best_epoch, no_improve = mpinn, opt_states, 0, 0
    lo, hi = 1.0 / max_weight, max_weight

    while total_epochs < max_epochs:
        block = min(block_epochs, max_epochs - total_epochs)

        mpinn, opt_states, block_h, _ = fit(
            mpinn,
            txs,
            opt_states,
            x_collocations,
            bc_configs,
            block,
            log_interval=log_interval,
            step_offset=total_epochs,
        )
        total_epochs += block

        for k, v in block_h.items():
            if k in history:
                history[k].extend(v)
            else:
                history[k] = v

        # --- обновление весов (PDE-anchored) ---
        last_aux = block_h["loss_components"][-1]
        l_pde = last_aux[0]
        if not math.isfinite(l_pde) or l_pde <= 0:
            l_pde = 1e-6

        raw = [
            last_aux[2] / l_pde,  # dir
            last_aux[3] / l_pde,  # neu
            last_aux[4] / l_pde,  # rob
            last_aux[5] / l_pde,  # if_T
            last_aux[6] / l_pde,  # if_flux
        ]

        w_old = [float(v) for v in mpinn.weights[1:]]
        w_new = [
            max(lo, min(hi, smoothing * w + (1.0 - smoothing) * r))
            for w, r in zip(w_old, raw)
        ]
        new_weights = jnp.asarray(
            [1.0] + w_new,
            dtype=mpinn.weights.dtype,
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

        print(
            f"[block {total_epochs:6d}] "
            f"{metric_key.upper()}={current_metric:.4e}  "
            f"w={[round(float(v), 3) for v in mpinn.weights]}  "
            f"(best={best_metric:.4e}, patience={no_improve}/{patience})"
        )

        if current_metric < target_metric:
            print(
                f"Target {metric_key.upper()}={target_metric:.2e} @ epoch {total_epochs}"
            )
            break
        if no_improve >= patience:
            print(f"Early stop: no progress for {patience} blocks")
            break

    history["epochs_trained"] = total_epochs
    history["best_epoch"] = best_epoch  # <--- ДОБАВИТЬ ЭТО
    return best_mpinn, best_states, history, time.perf_counter() - t0


# ============================================================ Wang et al., 2021
def compute_mpinn_grad_norms(mpinn, x_collocations, bc_configs, source_fns=()):
    """L-infinity нормы градиентов 6 компонентов: [pde, dir, neu, rob, if_T, if_flux]."""
    target_idx = [0, 2, 3, 4, 5, 6]
    norms = []
    for idx in target_idx:

        def closure(m, _idx=idx):
            _, aux = mpinn_loss_fn(m, x_collocations, bc_configs, source_fns)
            return aux[_idx]

        grads = eqx.filter_grad(closure)(mpinn)
        flat = jnp.concatenate(
            [
                jnp.ravel(g)
                for g in jax.tree_util.tree_leaves(grads)
                if eqx.is_inexact_array(g)
            ]
        )
        norms.append(float(jnp.linalg.norm(flat) + 1e-12))
    return norms


def fit_adaptive_weights_grad(
    mpinn,
    txs,
    opt_states,
    x_collocations,
    bc_configs,
    geometries,
    x_test,
    exact_fn,
    target_metric=1e-3,
    metric_key="rl2",
    block_epochs=500,
    max_epochs=20_000,
    log_interval=500,
    patience=10,
    smoothing=0.8,
    max_weight=5.0,
    source_fns=(),
):
    """Адаптивное обучение MPINN с фиксацией веса PDE (PDE-anchored)."""
    t0 = time.perf_counter()
    history = {"steps": [], "total_loss": [], "loss_components": [], "weights": []}
    total_epochs = 0
    best_metric, best_mpinn, best_states, best_epoch, no_improve = (
        float("inf"),
        mpinn,
        opt_states,
        0,
        0,
    )

    while total_epochs < max_epochs:
        block = min(block_epochs, max_epochs - total_epochs)
        # 1. Учимся один блок (fit внутри использует train_step)
        mpinn, opt_states, block_h, _ = fit(
            mpinn,
            txs,
            opt_states,
            x_collocations,
            bc_configs,
            block,
            log_interval=log_interval,
            step_offset=total_epochs,
            source_fns=source_fns,
        )
        total_epochs += block
        for k, v in block_h.items():
            history.setdefault(k, []).extend(v)
        # 2. Обновляем веса в КОНЦЕ КАЖДОГО БЛОКА
        norms = compute_mpinn_grad_norms(mpinn, x_collocations, bc_configs, source_fns)
        # --- ИЗМЕНЕНИЕ: Фиксация PDE как базы ---
        # norms[0] — это градиент PDE, так как в compute_mpinn_grad_norms
        # первым индексом идет PDE (target_idx = [0, 2, 3, 4, 5, 6])
        base_norm = norms[0]
        # Целевые веса: PDE получает 1.0, остальные растут относительно него
        raw = [base_norm / n for n in norms]
        # ----------------------------------------
        w_old = [float(v) for v in mpinn.weights]
        # EMA-сглаживание + клиппинг в [1/max_weight, max_weight]
        w_new = [
            max(
                1.0 / max_weight, min(max_weight, smoothing * w + (1.0 - smoothing) * r)
            )
            for w, r in zip(w_old, raw)
        ]
        mpinn = eqx.tree_at(lambda m: m.weights, mpinn, jnp.asarray(w_new))
        history["weights"].append(w_new)
        # 3. Оценка качества и Early Stopping
        metrics_dict, _, _ = evaluate(mpinn, geometries, x_test, exact_fn)
        current_metric = float(metrics_dict[metric_key])
        if current_metric < best_metric:
            best_metric, best_mpinn, best_states = current_metric, mpinn, opt_states
            best_epoch, no_improve = total_epochs, 0
        else:
            no_improve += 1
        print(
            f"[block {total_epochs:6d}] {metric_key.upper()}={current_metric:.4e} w={[round(float(v), 3) for v in mpinn.weights]} (best={best_metric:.4e}, patience={no_improve}/{patience})"
        )

        if current_metric < target_metric or no_improve >= patience:
            break

    history["epochs_trained"] = total_epochs
    history["best_epoch"] = best_epoch
    return best_mpinn, best_states, history, time.perf_counter() - t0


def multistage_adaptive_training(
    mpinn,
    txs,
    opt_states,
    x_collocations,
    bc_configs,
    geometries,
    x_test,
    exact_fn,
    optimizer_factory,
    init_opt_states_fn,
    fit_adaptive_fn,
    evaluate_fn,
    target_metric=1e-3,
    metric_key="rl2",
    block_epochs=500,
    max_epochs_per_restart=10_000,
    log_interval=1,
    initial_patience=100,
    lr_decay_factor=0.1,
    min_lr=1e-6,
    initial_lr=1e-3,
    verbose=True,
):
    """Multi-stage adaptive training with LR decay and restarts.

    Returns:
        best_mpinn, best_opt_states, history, total_time
    """
    MIN_PATIENCE = 10
    start_time = time.perf_counter()

    total_history = {
        "steps": [],
        "total_loss": [],
        "loss_components": [],
        "lr": [],
        "epochs_trained": 0,
        "best_epoch": 0,
    }

    # Инициализируем "лучшую" модель исходной
    best_mpinn = mpinn
    best_states = opt_states
    best_metric = float("inf")

    current_lr = initial_lr
    current_patience = initial_patience
    last_step = 0
    stage = 0

    while True:
        if verbose:
            print(
                f"STAGE #{stage} | LR = {current_lr:.2e} | "
                f"Patience = {current_patience}"
            )

        # ВАЖНО: передаем mpinn (который на первом этапе исходный,
        # а на последующих — лучший из предыдущего этапа)
        trained_mpinn, trained_states, hist, _ = fit_adaptive_fn(
            mpinn=mpinn,  # <-- ИСПОЛЬЗУЕМ ТЕКУЩУЮ (ЛУЧШУЮ) МОДЕЛЬ
            txs=txs,
            opt_states=opt_states,
            x_collocations=x_collocations,
            bc_configs=bc_configs,
            geometries=geometries,
            x_test=x_test,
            exact_fn=exact_fn,
            target_metric=target_metric,
            metric_key=metric_key,
            block_epochs=block_epochs,
            max_epochs=max_epochs_per_restart,
            log_interval=log_interval,
            patience=current_patience,
        )

        # Склейка истории
        local_steps = hist.get("steps", [])
        if local_steps and local_steps[0] == 0:
            shifted_steps = [s + last_step for s in local_steps]
            total_history["steps"].extend(shifted_steps)
            last_step = shifted_steps[-1] if shifted_steps else last_step
        else:
            total_history["steps"].extend(local_steps)
            if local_steps:
                last_step = local_steps[-1]

        for key in ("total_loss", "loss_components"):
            if key in hist:
                total_history[key].extend(hist[key])

        total_history["lr"].extend([current_lr] * len(local_steps))
        total_history["epochs_trained"] += len(local_steps)

        # Оценка
        metrics, _, _ = evaluate_fn(trained_mpinn, geometries, x_test, exact_fn)
        current_metric = float(metrics[metric_key])

        if verbose:
            print(f"Result: {metric_key.upper()} = {current_metric:.4e}")

        # Обновляем лучшую модель, если текущая лучше
        if current_metric < best_metric:
            best_metric = current_metric
            best_mpinn = trained_mpinn
            best_states = trained_states
            total_history["best_epoch"] = total_history["epochs_trained"]

        # Проверка выхода
        if current_metric < target_metric:
            if verbose:
                print(f"Target {target_metric:.2e} achieved.")
            break

        new_lr = current_lr * lr_decay_factor
        if new_lr < min_lr:
            if verbose:
                print(f"LR below minimum ({min_lr:.2e}). Stop.")
            break

        # Подготовка к следующему этапу
        current_lr = new_lr
        current_patience = max(MIN_PATIENCE, current_patience // 2)
        stage += 1

        # ВАЖНО: Для следующего этапа берем ЛУЧШУЮ модель,
        # а не просто trained_mpinn (которая может быть хуже).
        # Но так как мы уже сохранили best_mpinn,
        # передаем именно её.
        mpinn = best_mpinn

        # Пересоздаем оптимизаторы с новым LR и сбрасываем состояния
        txs = tuple(optimizer_factory(current_lr) for _ in range(len(txs)))
        opt_states = init_opt_states_fn(mpinn=mpinn, txs=txs)

    total_time = time.perf_counter() - start_time
    if verbose:
        print(f"Finished in {total_time:.1f}s, stages = {stage}")

    return best_mpinn, best_states, total_history, total_time
