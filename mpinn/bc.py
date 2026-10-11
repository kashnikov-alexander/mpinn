"""Boundary condition functions for 1D/2D/3D problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _broadcast(value, like):
    """Скаляр или массив -> массив формы like."""
    if jnp.ndim(value) == 0:
        return jnp.full_like(like, value)
    return jnp.ravel(value)


def _normal_derivative(model, points, normals):
    """dT/dn в точках. model ожидает (N, D) -> (N,)."""
    def grad_one(x):
        return jax.grad(lambda y: model(y[None]).ravel()[0])(x)

    grads = jax.vmap(grad_one)(points)
    return jnp.sum(grads * normals, axis=1)


def _lambda_vals(model, points, lam):
    """lam -> массив значений в точках. Поддерживает скаляр, массив, callable(T)."""
    if callable(lam):
        T_vals = model(points).ravel()
        return jax.vmap(lam)(T_vals)
    return lam * jnp.ones((points.shape[0],))


def dirichlet_bc(model, points, T_bc):
    """T = T_bc на границе."""
    T_pred = model(points).ravel()
    return T_pred - _broadcast(T_bc, T_pred)


def neumann_bc(model, points, normals, q, lam=1.0):
    """
    Закон Фурье: -lam * dT/dn = q.
    q    — тепловой поток вдоль ВНЕШНЕЙ нормали
    lam  — теплопроводность (скаляр, массив или callable(T))
    """
    lam_vals = _lambda_vals(model, points, lam)
    dT_dn = _normal_derivative(model, points, normals)
    return -lam_vals * dT_dn - _broadcast(q, dT_dn)


def robin_bc(model, points, normals, h, T_inf, lam=1.0):
    """
    Конвективный теплообмен:
        h * (T - T_inf) + lam * dT/dn = 0

    h     — коэффициент теплоотдачи
    T_inf — температура внешней среды
    lam   — теплопроводность (скаляр, массив или callable(T))
    """
    T_vals = model(points).ravel()
    lam_vals = _lambda_vals(model, points, lam)
    dT_dn = _normal_derivative(model, points, normals)
    return h * (T_vals - _broadcast(T_inf, T_vals)) + lam_vals * dT_dn


def adiabatic_bc(model, points, normals):
    """
    Адиабатическая граница: dT/dn = 0.
    Эквивалентно neumann_bc с q=0, но без параметра lam.
    """
    return _normal_derivative(model, points, normals)