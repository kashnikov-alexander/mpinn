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


# ==================== 2D задачи ====================

def laplace_2d(model, x, lam, source_fn=0.0):
    """lam(T)*(T_xx + T_yy) + lam'(T)*|grad T|**2 + f(x,y) = 0"""
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


def polar_2d(model, x, lam, source_fn=0.0):
    """
    Радиально-симметричная задача в полярных координатах (только от r):
        lam(T)*(T'' + T'/r) + lam'(T)*(T')**2 + f(r) = 0

    Производные по r берутся при theta = 0; корректно, если решение
    действительно не зависит от угла.
    """
    r = x[:, 0]

    def predict_radial(r_val):
        pt = jnp.array([[r_val, 0.0]])
        return model(pt).ravel()[0]

    grad_T = jax.grad(predict_radial)
    hess_T = jax.grad(grad_T)

    r_safe = jnp.where(r == 0.0, 1e-12, r)

    dT_dr = jax.vmap(grad_T)(r)
    d2T_dr2 = jax.vmap(hess_T)(r)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    laplacian = d2T_dr2 + (1.0 / r_safe) * dT_dr
    residual = lambda_vals * laplacian + dlambda_dT * dT_dr ** 2 + _source(x, source_fn)
    return residual


def sphere_2d(model, x, lam, source_fn=0.0):
    """
    Сферический Лаплас в 2D-декартовых (x, y) с 3D-сферической симметрией:
        T_xx + T_yy + (x*T_x + y*T_y)/r**2

    Полное уравнение:
        lam(T)*[T_xx + T_yy + (x*T_x + y*T_y)/r**2]
        + lam'(T)*|grad T|**2 + f = 0
    """
    def predict(x_pt):
        return model(jnp.atleast_2d(x_pt)).ravel()[0]

    grad_fn = jax.grad(predict)
    hess_fn = jax.hessian(predict)

    def laplacian_pt(x_pt):
        xv, yv = x_pt[0], x_pt[1]
        r = jnp.sqrt(xv ** 2 + yv ** 2)
        r_safe = jnp.where(r == 0.0, 1e-12, r)

        H = hess_fn(x_pt)
        g = grad_fn(x_pt)

        lap_2d = H[0, 0] + H[1, 1]
        extra = (xv * g[0] + yv * g[1]) / (r_safe ** 2)
        return lap_2d + extra

    laplacian = jax.vmap(laplacian_pt)(x)
    grad_vals = jax.vmap(grad_fn)(x)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    grad_sq = jnp.sum(grad_vals ** 2, axis=1)
    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + _source(x, source_fn)
    return residual


# ==================== 3D задачи ====================

def laplace_3d(model, x, lam, source_fn=0.0):
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


def cylinder_3d_axisymmetric(model, x, lam, source_fn=0.0):
    """
    Осесимметричный цилиндр, координаты (r, z):
        lam(T)*(T_rr + T_r/r + T_zz) + lam'(T)*((T_r)**2 + (T_z)**2) + f(r,z) = 0
    """
    def predict(rz):
        return model(rz).ravel()[0]

    grad_fn = jax.grad(predict)
    hess_fn = jax.hessian(predict)

    def laplacian_pt(rz_pt):
        g = grad_fn(rz_pt)
        h = hess_fn(rz_pt)
        r = rz_pt[0]
        r_safe = jnp.where(r == 0.0, 1e-12, r)
        return h[0, 0] + (1.0 / r_safe) * g[0] + h[1, 1]

    laplacian = jax.vmap(laplacian_pt)(x)
    grad_vals = jax.vmap(grad_fn)(x)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    grad_sq = jnp.sum(grad_vals ** 2, axis=1)
    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + _source(x, source_fn)
    return residual


def sphere_3d_axisymmetric(model, x, lam, source_fn=0.0):
    """
    Осесимметричная сфера, координаты (r, theta), theta in (0, pi):
        lam(T)*(T_rr + 2*T_r/r + T_tt/r**2 + cot(theta)*T_t/r**2)
        + lam'(T)*((T_r)**2 + (T_t)**2/r**2) + f(r, theta) = 0
    """
    def predict(rtheta):
        return model(rtheta).ravel()[0]

    grad_fn = jax.grad(predict)
    hess_fn = jax.hessian(predict)

    def laplacian_pt(rtheta_pt):
        g = grad_fn(rtheta_pt)
        h = hess_fn(rtheta_pt)

        r = rtheta_pt[0]
        theta = rtheta_pt[1]

        r_safe = jnp.where(r == 0.0, 1e-12, r)
        sin_theta = jnp.sin(theta)
        sin_safe = jnp.where(jnp.abs(sin_theta) < 1e-12, 1e-12, sin_theta)
        cot_theta = jnp.cos(theta) / sin_safe

        T_rr = h[0, 0]
        T_r = g[0]
        T_tt = h[1, 1]
        T_t = g[1]

        return (
            T_rr
            + (2.0 / r_safe) * T_r
            + (1.0 / (r_safe ** 2)) * T_tt
            + (cot_theta / (r_safe ** 2)) * T_t
        )

    laplacian = jax.vmap(laplacian_pt)(x)
    grad_vals = jax.vmap(grad_fn)(x)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    # |grad T|**2 = (T_r)**2 + (T_t)**2 / r**2
    r_vals = x[:, 0]
    r_safe = jnp.where(r_vals == 0.0, 1e-12, r_vals)
    grad_sq = grad_vals[:, 0] ** 2 + grad_vals[:, 1] ** 2 / (r_safe ** 2)

    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + _source(x, source_fn)
    return residual