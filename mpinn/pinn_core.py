"""
Ядро реализации PINN на основе JAX и Equinox.
Поддерживает 1D, 2D и 3D задачи.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from mpinn.config import normalize_coords, denormalize_temp


class FCNet(eqx.Module):
    """Полносвязная нейронная сеть для PINN."""

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

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.linear_out(x)


class ScaledNet(eqx.Module):
    """Обёртка: нормализует вход, денормализует выход (T = T_norm * T_max)."""

    base: eqx.Module
    x_min: jax.Array = eqx.field(static=True)
    x_max: jax.Array = eqx.field(static=True)
    T_max: float = eqx.field(static=True)

    def __init__(self, base, x_min, x_max, T_max):
        self.base = base
        self.x_min = jnp.asarray(x_min)
        self.x_max = jnp.asarray(x_max)
        self.T_max = float(T_max)

    def __call__(self, x):
        x_norm = normalize_coords(x, self.x_min, self.x_max)
        return denormalize_temp(self.base(x_norm), self.T_max)



def _freeze_bc_configs(bc_configs: list[dict]) -> tuple:
    """
    list[dict] → tuple из (fn, points, normals, params_tuple).
    Нужно, чтобы bc_configs можно было держать как static-поле eqx.Module.
    """
    frozen = []
    for c in bc_configs:
        params = tuple(sorted(c.get("params", {}).items()))
        frozen.append((
            c["fn"],
            jnp.asarray(c["points"]),
            jnp.asarray(c["normals"]) if c.get("normals") is not None else None,
            params,
        ))
    return tuple(frozen)


class PINN(eqx.Module):
    """Физически-информированная нейронная сеть (Equinox + Optax)."""

    net: eqx.Module

    tx: optax.GradientTransformation = eqx.field(static=True)
    pde_fn: Callable = eqx.field(static=True)
    bc_configs: tuple = eqx.field(static=True)
    lam: Any = eqx.field(static=True)
    source_fn: Any = eqx.field(static=True)
    weights: tuple = eqx.field(static=True)

    def __init__(self, net, opt, weights, pde_fn, bc_configs, lam, source_fn=0.0):
        self.net = net
        self.tx = opt
        self.weights = tuple(weights)
        self.pde_fn = pde_fn
        self.bc_configs = _freeze_bc_configs(bc_configs)
        self.lam = lam
        self.source_fn = source_fn

    # ------------------------------------------------------------ forward
    def __call__(self, x):
        return self.net(x)

    def predict(self, x_test):
        return self.net(x_test)

    # ------------------------------------------------------------ helpers
    def init_opt_state(self):
        params = eqx.filter(self.net, eqx.is_inexact_array)
        return self.tx.init(params)

    def set_lam(self, new_lam):
        """
        Возвращает новый PINN с обновлённой теплопроводностью.
        Так как lam — static-поле, jit пересоберётся автоматически
        при изменении значения.
        """
        return eqx.tree_at(lambda p: p.lam, self, new_lam)

    # ------------------------------------------------------------ metrics
    def compute_metrics(self, x_test, T_pred, T_exact):
        diff = T_pred - T_exact
        mse = float(jnp.mean(diff ** 2))
        mae = float(jnp.mean(jnp.abs(diff)))
        rmse = float(jnp.sqrt(mse))
        max_error = float(jnp.max(jnp.abs(diff)))
        mape = float(jnp.mean(jnp.abs(diff / (jnp.abs(T_exact) + 1e-8))))
        return {
            "mape":      f"{mape:.4e}",
            "mae":       f"{mae:.4e}",
            "mse":       f"{mse:.4e}",
            "rmse":      f"{rmse:.4e}",
            "max_error": f"{max_error:.4e}",
        }

    def evaluate(self, x_test, exact_fn, bc_info=None):
        T_pred = self.predict(x_test).ravel()
        T_exact = exact_fn(x_test).ravel()
        metrics = self.compute_metrics(x_test, T_pred, T_exact)
        if bc_info is not None:
            for name, bc_type in bc_info.items():
                metrics[f"bc_{name}"] = bc_type
        return metrics, T_pred, T_exact

    # ------------------------------------------------------------ fit
    def fit(self, x_collocation, epochs, log_interval=100, print_log=False):
        start_time = time.perf_counter()

        pinn = self
        opt_state = self.init_opt_state()

        n_bc = len(self.bc_configs)
        loss_names = ["pde", "bc_total"] + [f"bc_{i}" for i in range(n_bc)]
        history = {"total_loss": []}
        for name in loss_names:
            history[name] = []

        for step in range(epochs):
            pinn, opt_state, total_loss, aux_losses = train_step(
                pinn, opt_state, x_collocation
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

        # записать веса обратно (Python-мутация, не PyTree-операция)
        self.net = pinn.net
        return history, time.perf_counter() - start_time

    # ------------------------------------------------------------ adaptive
    def _snapshot_pinn(self):
        return jax.tree.map(jnp.copy, self)

    def _make_rl2_evaluator(self, x_test, exact_fn):
        @eqx.filter_jit
        def _rl2(pinn):
            T_pred = pinn.predict(x_test).ravel()
            T_exact = exact_fn(x_test).ravel()
            return jnp.linalg.norm(T_pred - T_exact) / (
                jnp.linalg.norm(T_exact) + 1e-12
            )
        return _rl2

    def _make_unweighted_loss_evaluator(self, x_collocation):
        @eqx.filter_jit
        def _loss(pinn):
            total, (pde, *bcs) = loss_fn(pinn, x_collocation)
            bc_total = sum(bcs) if bcs else 0.0
            return pde + bc_total
        return _loss

    def fit_adaptive(
        self,
        x_collocation,
        x_test,
        exact_fn,
        target_rl2: float = 1e-3,
        block_epochs: int = 50,
        max_epochs: int = 5000,
        log_interval: int = 100,
    ):
        start_time = time.perf_counter()

        rl2_fn = self._make_rl2_evaluator(x_test, exact_fn)
        history: dict[str, list] = {}
        total_epochs = 0
        best_rl2 = float("inf")
        best_snapshot = self._snapshot_pinn()

        while total_epochs < max_epochs:
            block = min(block_epochs, max_epochs - total_epochs)
            block_history, _ = self.fit(x_collocation, block, log_interval)
            total_epochs += block

            for key, values in block_history.items():
                history.setdefault(key, []).extend(values)

            rl2 = float(rl2_fn(self))
            if rl2 < best_rl2:
                best_rl2 = rl2
                best_snapshot = self._snapshot_pinn()

            if rl2 < target_rl2:
                break

        # восстановить лучший снапшот
        self.net = best_snapshot.net
        return history, time.perf_counter() - start_time

    def fit_loss(
        self,
        x_collocation,
        epochs: int,
        log_interval: int = 100,
        checkpoint_every: int = 50,
    ):
        start_time = time.perf_counter()

        loss_eval = self._make_unweighted_loss_evaluator(x_collocation)
        history: dict[str, list] = {}
        total_epochs = 0
        best_loss = float("inf")
        best_snapshot = self._snapshot_pinn()

        while total_epochs < epochs:
            block = min(checkpoint_every, epochs - total_epochs)
            block_history, _ = self.fit(x_collocation, block, log_interval)
            total_epochs += block

            for key, values in block_history.items():
                history.setdefault(key, []).extend(values)

            loss = float(loss_eval(self))
            if loss < best_loss:
                best_loss = loss
                best_snapshot = self._snapshot_pinn()

        self.net = best_snapshot.net
        return history, time.perf_counter() - start_time
    

def loss_fn(pinn: PINN, x_collocation):
    """Чистая функция лосса. Возвращает (total, (loss_pde, *loss_bcs))."""
    model = pinn.net
    residuals_pde = pinn.pde_fn(model, x_collocation, pinn.lam, pinn.source_fn)
    loss_pde = jnp.mean(residuals_pde ** 2)

    loss_bcs = []
    for fn, points, normals, params_tuple in pinn.bc_configs:
        params = dict(params_tuple)
        if normals is not None:
            residuals = fn(model, points, normals, **params)
        else:
            residuals = fn(model, points, **params)
        loss_bcs.append(jnp.mean(residuals ** 2))

    loss_bc_total = sum(loss_bcs) if loss_bcs else jnp.array(0.0)
    total = pinn.weights[0] * loss_pde + pinn.weights[1] * loss_bc_total
    return total, (loss_pde, *loss_bcs)


@eqx.filter_jit
def train_step(pinn: PINN, opt_state, x_collocation):
    """
    Один шаг обучения. Возвращает (new_pinn, new_opt_state, total, aux).
    Обновляются только inexact-array-листья (веса сети).
    """
    def closure(p):
        return loss_fn(p, x_collocation)

    (total, aux), grads = eqx.filter_value_and_grad(closure, has_aux=True)(pinn)

    params = eqx.filter(pinn, eqx.is_inexact_array)
    updates, new_opt_state = pinn.tx.update(grads, opt_state, params)

    new_pinn = eqx.apply_updates(pinn, updates)
    return new_pinn, new_opt_state, total, aux