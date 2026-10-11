"""
MPINN: мультидоменная PINN на Equinox в функциональном стиле.
Контейнер — только поля. Всё остальное — свободные функции.
Веса передаются аргументами в loss_fn (разные задачи — разные лоссы).
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp

from .pinn_core import PINN
from .pinn_core import loss_fn as pinn_loss_fn, compute_metrics


class MPINN(eqx.Module):
    """Только поля: список PINN + геометрия интерфейсов."""

    pinns: tuple
    interface_pts: tuple
    interface_n1: tuple
    interface_n2: tuple
    interface_pairs: tuple = eqx.field(static=True)

    def __init__(self, pinns, interface_data):
        self.pinns = tuple(pinns)
        self.interface_pts = tuple(d[0] for d in interface_data)
        self.interface_n1 = tuple(d[1] for d in interface_data)
        self.interface_n2 = tuple(d[2] for d in interface_data)
        self.interface_pairs = tuple((int(d[3]), int(d[4])) for d in interface_data)


def build_mpinn(domain_configs, interface_pairs, n_interface_points, rng=None):
    """Собирает MPINN и точки коллокации из конфигов доменов."""
    if rng is None:
        rng = jax.random.PRNGKey(0)

    pinns = tuple(
        PINN(
            net=cfg["net"],
            pde_fn=cfg["pde_fn"],
            bc_configs=cfg["bc_configs"],
            lam=cfg.get("lam", 1.0),
        )
        for cfg in domain_configs
    )

    col_keys = jax.random.split(rng, len(domain_configs))
    x_collocations = tuple(
        cfg["geom"].sample_interior(cfg["n_points"], rng=k)
        for cfg, k in zip(domain_configs, col_keys)
    )

    itf_keys = jax.random.split(rng, max(len(interface_pairs), 1))
    interface_data = []
    for (idx1, name1, idx2, name2), k in zip(interface_pairs, itf_keys):
        g1, g2 = domain_configs[idx1]["geom"], domain_configs[idx2]["geom"]
        pts, n1, n2 = g1.sample_interface(
            other_geom=g2,
            n_points=n_interface_points,
            self_interface_name=name1,
            other_interface_name=name2,
            rng=k,
        )
        interface_data.append((pts, n1, n2, idx1, idx2))

    return MPINN(pinns=pinns, interface_data=interface_data), x_collocations


def interface_loss(model_i, model_j, pts, n1, n2, lam_i, lam_j):
    """continuity_T + continuity_flux на одном интерфейсе."""
    T1 = model_i(pts).ravel()
    T2 = model_j(pts).ravel()
    cont_T = jnp.mean((T1 - T2) ** 2)

    def dT_dn(model, x, n):
        return jax.jvp(lambda y: model(y[None]).ravel()[0], (x,), (n,))[1]

    dT1n = jax.vmap(dT_dn, in_axes=(None, 0, 0))(model_i, pts, n1)
    dT2n = jax.vmap(dT_dn, in_axes=(None, 0, 0))(model_j, pts, n2)
    cont_flux = jnp.mean((lam_i * dT1n + lam_j * dT2n) ** 2)

    return cont_T + cont_flux


def mpinn_loss_fn(mpinn, x_collocations, *,
                  pde_weight=1.0, bc_weight=1.0, if_weight=1.0,
                  source_fns=()):
    """
    Три глобальных лосса MPINN:
        pde_total = Σ_domains loss_pde_i              (сырые, из aux)
        bc_total  = Σ_domains Σ_bc loss_bc_i          (сырые, из aux)
        if_total  = Σ_interfaces interface_loss_j

    total = pde_weight · pde_total
          + bc_weight  · bc_total
          + if_weight  · if_total
    """
    if not source_fns:
        source_fns = (0.0,) * len(mpinn.pinns)

    pde_total = jnp.array(0.0)
    bc_total  = jnp.array(0.0)
    for pinn, x_d, src in zip(mpinn.pinns, x_collocations, source_fns):
        _, (pde_raw, *bcs_raw) = pinn_loss_fn(pinn, x_d, src)
        pde_total = pde_total + pde_raw
        if bcs_raw:
            bc_total = bc_total + sum(bcs_raw)

    if_total = jnp.array(0.0)
    for pts, n1, n2, (i, j) in zip(
        mpinn.interface_pts, mpinn.interface_n1, mpinn.interface_n2,
        mpinn.interface_pairs,
    ):
        p_i, p_j = mpinn.pinns[i], mpinn.pinns[j]
        if_total = if_total + interface_loss(
            p_i.net, p_j.net, pts, n1, n2, p_i.lam, p_j.lam
        )

    total = (pde_weight * pde_total
             + bc_weight * bc_total
             + if_weight * if_total)
    return total, (pde_total, bc_total, if_total)


def init_opt_states(mpinn, txs):
    return tuple(
        tx.init(eqx.filter(p, eqx.is_inexact_array))
        for tx, p in zip(txs, mpinn.pinns)
    )


@eqx.filter_jit
def train_step(mpinn, txs, opt_states, x_collocations, loss_kwargs):
    """loss_kwargs — dict с весами, пробрасывается в mpinn_loss_fn."""
    def closure(m):
        return mpinn_loss_fn(m, x_collocations, **loss_kwargs)

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


def fit(mpinn, txs, opt_states, x_collocations, epochs,
        loss_kwargs=None, log_interval=100, print_log=False):
    loss_kwargs = loss_kwargs or {}
    history = {k: [] for k in ("total_loss", "pde_total", "bc_total", "if_total")}
    t0 = time.perf_counter()

    for step in range(epochs):
        mpinn, opt_states, total, aux = train_step(
            mpinn, txs, opt_states, x_collocations, loss_kwargs
        )
        if step % log_interval == 0 or step == epochs - 1:
            history["total_loss"].append(float(total))
            history["pde_total"].append(float(aux[0]))
            history["bc_total"].append(float(aux[1]))
            history["if_total"].append(float(aux[2]))
            if print_log:
                print(f"\rstep {step:6d} | total={float(total):.3e} "
                      f"pde={float(aux[0]):.3e} bc={float(aux[1]):.3e} "
                      f"if={float(aux[2]):.3e}".ljust(120), end="")
    if print_log:
        print()

    return mpinn, opt_states, history, time.perf_counter() - t0


def predict(mpinn, geometries, x_test, dim=None):
    """Предсказание по доменам с маскировкой. Первый домен побеждает при пересечении."""
    x = jnp.atleast_1d(x_test)
    if x.ndim == 1:
        dim = dim or getattr(geometries[0], "dim", x.shape[-1])
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


def evaluate(mpinn, geometries, x_test, exact_fn, dim=None):
    t_pred = predict(mpinn, geometries, x_test, dim=dim).ravel()
    t_exact = exact_fn(x_test).ravel()
    return compute_metrics(t_pred, t_exact), t_pred, t_exact


def rl2_error(mpinn, geometries, x_test, exact_fn, dim=None):
    t_pred = predict(mpinn, geometries, x_test, dim=dim).ravel()
    t_exact = exact_fn(x_test).ravel()
    return jnp.linalg.norm(t_pred - t_exact) / (jnp.linalg.norm(t_exact) + 1e-12)


def fit_adaptive(mpinn, txs, opt_states, x_collocations, geometries,
                 x_test, exact_fn,
                 target_rl2=1e-3, block_epochs=500, max_epochs=50_000,
                 log_interval=500, patience=5, tol=1e-6,
                 loss_kwargs=None, dim=None):
    loss_kwargs = loss_kwargs or {}
    t0 = time.perf_counter()
    history = {}
    total_epochs = best_rl2 = 0
    best_rl2 = float("inf")
    best_mpinn, best_states, best_epoch, no_improve = mpinn, opt_states, 0, 0

    while total_epochs < max_epochs:
        block = min(block_epochs, max_epochs - total_epochs)
        mpinn, opt_states, block_h, _ = fit(
            mpinn, txs, opt_states, x_collocations, block,
            loss_kwargs=loss_kwargs, log_interval=log_interval,
        )
        total_epochs += block
        for k, v in block_h.items():
            history.setdefault(k, []).extend(v)

        rl2 = float(rl2_error(mpinn, geometries, x_test, exact_fn, dim=dim))
        if rl2 < best_rl2 - tol:
            best_rl2, best_mpinn, best_states = rl2, mpinn, opt_states
            best_epoch, no_improve = total_epochs, 0
        else:
            no_improve += 1

        print(f"[block {total_epochs:6d}] RL2={rl2:.4e} "
              f"(best={best_rl2:.4e} @ {best_epoch}, "
              f"patience={no_improve}/{patience})")

        if rl2 < target_rl2:
            print(f"Target RL2={target_rl2:.2e} @ epoch {total_epochs}")
            break
        if no_improve >= patience:
            print(f"Early stop: no progress for {patience} blocks")
            break

    history['epochs_trained'] = total_epochs
    return best_mpinn, best_states, history, time.perf_counter() - t0