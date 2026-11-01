"""
Функции невязки PDE для задач теплопроводности.
Все функции возвращают массив невязок (N,).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _lambda_and_derivative(T_pred, lam):
    """Возвращает (lambda(T), dlambda/dT). lam — скаляр или callable(T)."""
    if callable(lam):
        lambda_vals = jax.vmap(lam)(T_pred)
        dlambda_dT = jax.vmap(jax.grad(lam))(T_pred)
    else:
        lambda_vals = lam * jnp.ones_like(T_pred)
        dlambda_dT = jnp.zeros_like(T_pred)
    return lambda_vals, dlambda_dT


def _source(x, source_fn):
    """source_fn — скаляр (обычно 0.0) или callable(x)."""
    if callable(source_fn):
        return source_fn(x)
    return source_fn


# ==================== 1D задачи ====================

def line_1d(model, x, lam, source_fn=0.0):
    """lam(T)*T'' + lam'(T)*(T')**2 + f(x) = 0"""
    def predict(x_val):
        return model(jnp.atleast_2d(x_val)).ravel()[0]

    grad_T = jax.grad(predict)
    dT_dx = jax.vmap(grad_T)(x.ravel())
    d2T_dx2 = jax.vmap(jax.grad(grad_T))(x.ravel())

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    residual = lambda_vals * d2T_dx2 + dlambda_dT * dT_dx ** 2 + _source(x.ravel(), source_fn)
    return residual


def cylinder_1d(model, x, lam, source_fn=0.0):
    """lam(T)*(T'' + T'/r) + lam'(T)*(T')**2 + f(r) = 0"""
    def predict(r_val):
        return model(jnp.atleast_2d(r_val)).ravel()[0]

    grad_T = jax.grad(predict)
    hess_T = jax.grad(grad_T)

    r = x.ravel()
    r_safe = jnp.where(r == 0.0, 1e-12, r)

    dT_dr = jax.vmap(grad_T)(r)
    d2T_dr2 = jax.vmap(hess_T)(r)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    laplacian = d2T_dr2 + (1.0 / r_safe) * dT_dr
    residual = lambda_vals * laplacian + dlambda_dT * dT_dr ** 2 + _source(r, source_fn)
    return residual


def sphere_1d(model, x, lam, source_fn=0.0):
    """lam(T)*(T'' + 2*T'/r) + lam'(T)*(T')**2 + f(r) = 0"""
    def predict(r_val):
        return model(jnp.atleast_2d(r_val)).ravel()[0]

    grad_T = jax.grad(predict)
    hess_T = jax.grad(grad_T)

    r = x.ravel()
    r_safe = jnp.where(r == 0.0, 1e-12, r)

    dT_dr = jax.vmap(grad_T)(r)
    d2T_dr2 = jax.vmap(hess_T)(r)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    laplacian = d2T_dr2 + (2.0 / r_safe) * dT_dr
    residual = lambda_vals * laplacian + dlambda_dT * dT_dr ** 2 + _source(r, source_fn)
    return residual


# ==================== Универсальный оператор (2D, 3D и т.д.) ====================
def laplace(model, x, lam, source_fn=0.0):
    """lam(T)*Laplacian(T) + lam'(T)*|grad T|**2 + f(x,y,z) = 0"""
    def predict(x_pts):
        return model(jnp.atleast_2d(x_pts)).ravel()[0]

    def laplacian_pt(x_pt):
        return jnp.trace(jax.hessian(predict)(x_pt))

    laplacian = jax.vmap(laplacian_pt)(x)
    grad_vals = jax.vmap(jax.grad(predict))(x)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    grad_sq = jnp.sum(grad_vals ** 2, axis=1)
    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + _source(x, source_fn)
    return residual