"""
Точные аналитические решения для 1D задач теплопроводности.

Конвенции:
    - x0 / r0 — граница Дирихле: T = T0.
    - x1 / r1 — вторая граница (Нейман или Робин).
    - Знак потока: -lam * dT/dn = q (закон Фурье, q вдоль внешней нормали).

Каждая фабрика возвращает вызываемый объект exact(x).
"""

import jax.numpy as jnp


# ==================== плоская стенка ====================

def line_1d_dirichlet_exact(T0, T1, x0, x1):
    """T(x0)=T0, T(x1)=T1."""
    def exact(x):
        return T0 + (T1 - T0) * (x - x0) / (x1 - x0)
    return exact


def line_1d_neumann_exact(T0, q, x0, x1, lam):
    """T(x0)=T0, -lam*dT/dx|_{x1}=q."""
    grad = -q / lam

    def exact(x):
        return T0 + grad * (x - x0)
    return exact


def line_1d_robin_exact(T0, T_inf, h, x0, x1, lam):
    """T(x0)=T0, h*(T(x1)-T_inf) + lam*dT/dx|_{x1}=0."""
    A = h * (T_inf - T0) / (h * (x1 - x0) + lam)

    def exact(x):
        return T0 + A * (x - x0)
    return exact


# ==================== цилиндр ====================

def cylinder_1d_dirichlet_exact(T0, T1, r0, r1):
    """T(r0)=T0, T(r1)=T1."""
    def exact(x):
        return T0 + (T1 - T0) * jnp.log(x / r0) / jnp.log(r1 / r0)
    return exact


def cylinder_1d_neumann_exact(T0, q, r0, r1, lam):
    """T(r0)=T0, -lam*dT/dr|_{r1}=q."""
    C = (-q / lam) * r1

    def exact(x):
        return T0 + C * jnp.log(x / r0)
    return exact


def cylinder_1d_robin_exact(T0, T_inf, h, r0, r1, lam):
    """T(r0)=T0, h*(T(r1)-T_inf) + lam*dT/dr|_{r1}=0."""
    C1 = h * r1 * (T_inf - T0) / (lam + h * r1 * jnp.log(r1 / r0))

    def exact(x):
        return T0 + C1 * jnp.log(x / r0)
    return exact


# ==================== сфера ====================

def sphere_1d_dirichlet_exact(T0, T1, r0, r1):
    """T(r0)=T0, T(r1)=T1."""
    inv_r0 = 1.0 / r0
    inv_r1 = 1.0 / r1

    def exact(x):
        return T0 + (T1 - T0) * (inv_r0 - 1.0 / x) / (inv_r0 - inv_r1)
    return exact


def sphere_1d_neumann_exact(T0, q, r0, r1, lam):
    """T(r0)=T0, -lam*dT/dr|_{r1}=q."""
    grad = -q / lam

    def exact(x):
        return T0 - grad * r1 ** 2 * (1.0 / x - 1.0 / r0)
    return exact


def sphere_1d_robin_exact(T0, T_inf, h, r0, r1, lam):
    """T(r0)=T0, h*(T(r1)-T_inf) + lam*dT/dr|_{r1}=0."""
    C1 = h * (T_inf - T0) / ((lam / r1 ** 2) + h * (1.0 / r0 - 1.0 / r1))

    def exact(x):
        return T0 + C1 * (1.0 / r0 - 1.0 / x)
    return exact