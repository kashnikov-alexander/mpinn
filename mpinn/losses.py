"""Функции потерь для PINN.

Два стандартных варианта:
    - loss_fn         — все BC в одну кучу; weights = [w_pde, w_bc].
    - loss_fn_split_bc — BC по типам; weights = [w_pde, w_dir, w_neu, w_rob].

Контракт: loss_fn(pinn, x_collocation, bc_configs, source_fn=0.0)
          -> (total, aux)
    aux: tuple (classic) или jnp.array (split) — fit() пишет по ним history.

Кастомные лоссы с другими весами кладутся сюда же; контракт по len(weights)
и по ключам history задаёт сама функция.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from .bc import DirichletBC, NeumannBC, RobinBC, AdiabaticBC
if TYPE_CHECKING:
    from .pinn_core import PINN


# ============================================================ лоссы

def loss_fn(pinn: PINN, x_collocation, bc_configs, source_fn=0.0):
    """Классическая функция потерь: все BC в одну кучу.

    Требует len(pinn.weights) == 2: [w_pde, w_bc].
    Возвращает (total, (loss_pde, *loss_bcs)).
    """
    residuals_pde = pinn.pde_fn(pinn.net, x_collocation, pinn.lam, source_fn)
    loss_pde = jnp.mean(residuals_pde ** 2)

    loss_bcs = [
        jnp.mean(bc.residual(pinn.net, pinn.lam) ** 2)
        for bc in bc_configs
    ]
    loss_bc_total = sum(loss_bcs) if loss_bcs else jnp.array(0.0)
    total = pinn.weights[0] * loss_pde + pinn.weights[1] * loss_bc_total
    return total, (loss_pde, *loss_bcs)


def loss_fn_split_bc(pinn: PINN, x_collocation, bc_configs, source_fn=0.0):
    """BC разделены на 3 типа.

    Требует len(pinn.weights) == 4: [w_pde, w_dir, w_neu, w_rob].
    AdiabaticBC учитывается как Neumann.
    Возвращает (total, jnp.array([loss_pde, loss_dir, loss_neu, loss_rob])).
    """
    residuals_pde = pinn.pde_fn(pinn.net, x_collocation, pinn.lam, source_fn)
    loss_pde = jnp.mean(residuals_pde ** 2)

    loss_dir = jnp.array(0.0)
    loss_neu = jnp.array(0.0)
    loss_rob = jnp.array(0.0)

    for bc in bc_configs:
        val = jnp.mean(bc.residual(pinn.net, pinn.lam) ** 2)
        if isinstance(bc, DirichletBC):
            loss_dir = loss_dir + val
        elif isinstance(bc, (NeumannBC, AdiabaticBC)):
            loss_neu = loss_neu + val
        elif isinstance(bc, RobinBC):
            loss_rob = loss_rob + val
        else:
            raise TypeError(f"Unknown BC type: {type(bc).__name__}")

    losses_vec = jnp.array([loss_pde, loss_dir, loss_neu, loss_rob])
    total = jnp.dot(pinn.weights, losses_vec)
    return total, losses_vec