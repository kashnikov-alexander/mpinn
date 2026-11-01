"""Boundary condition classes for 1D/2D/3D problems."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


# ============================================================ helpers

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


# ============================================================ BC hierarchy

class BC(eqx.Module):
    """Базовый класс граничного условия."""
    points: jax.Array

    def residual(self, net, lam):
        raise NotImplementedError


class DirichletBC(BC):
    """T = T на границе."""
    T: float | jax.Array

    def residual(self, net, lam):
        T_pred = net(self.points).ravel()
        return T_pred - _broadcast(self.T, T_pred)


class NeumannBC(BC):
    """Закон Фурье: -lam * dT/dn = q."""
    normals: jax.Array
    q: float | jax.Array

    def residual(self, net, lam):
        dT_dn = _normal_derivative(net, self.points, self.normals)
        lam_vals = _lambda_vals(net, self.points, lam)
        return -lam_vals * dT_dn - _broadcast(self.q, dT_dn)


class RobinBC(BC):
    """Конвективный теплообмен: h * (T - T_inf) + lam * dT/dn = 0."""
    normals: jax.Array
    h: float
    T_inf: float

    def residual(self, net, lam):
        T_vals = net(self.points).ravel()
        lam_vals = _lambda_vals(net, self.points, lam)
        dT_dn = _normal_derivative(net, self.points, self.normals)
        return self.h * (T_vals - _broadcast(self.T_inf, T_vals)) + lam_vals * dT_dn


class AdiabaticBC(BC):
    """Адиабатическая граница: dT/dn = 0."""
    normals: jax.Array

    def residual(self, net, lam):
        return _normal_derivative(net, self.points, self.normals)