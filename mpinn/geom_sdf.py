"""SDF-геометрия с поддержкой CSG и именованных граней.

Поддерживает: Box2D, Circle.
Операции: union (+), difference (-), intersection (&).
Грани именуются автоматически или задаются пользователем.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .geom import GeometryBase

# ============================================================ SDF-примитивы (чистые функции)


def sdf_box(x, lo, hi):
    """SDF для прямоугольника (2D или 3D). x: (N, D)"""
    # Центр и половинные размеры
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    q = jnp.abs(x - center) - half
    # Внутри (q < 0) и снаружи (q > 0)
    return jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) + jnp.minimum(
        jnp.max(q, axis=-1), 0.0
    )


def sdf_circle(x, c, r):
    """SDF для круга. x: (N, 2)"""
    return jnp.linalg.norm(x - c, axis=-1) - r


# ============================================================ CSG-операции


def csg_union(a, b):
    return jnp.minimum(a, b)


def csg_intersection(a, b):
    return jnp.maximum(a, b)


def csg_difference(a, b):
    return jnp.maximum(a, -b)


# ============================================================ SDFGeometry


class SDFGeometry(GeometryBase):
    """Геометрия на основе SDF с поддержкой CSG."""

    sdf_fn: callable = eqx.field(static=True)
    bounds: tuple = eqx.field(static=True)
    children: tuple = eqx.field(static=True)  # (child_a, child_b, op)

    primitive_name: str = eqx.field(static=True)
    primitive_data: dict = eqx.field(static=True)

    def __init__(
        self, sdf_fn, bounds, primitive_name=None, primitive_data=None, children=()
    ):
        super().__init__(dim=len(bounds[0]))
        self.sdf_fn = sdf_fn
        self.bounds = bounds
        self.primitive_name = primitive_name
        self.primitive_data = primitive_data or {}
        self.children = children

    def is_inside(self, x):
        return self.sdf_fn(x) < 0

    @property
    def min_corner(self):
        return self.bounds[0]

    @property
    def max_corner(self):
        return self.bounds[1]

    def _primitive_boundary(self, n, rng):
        """Возвращает словарь {name: (pts, norms)} для листового примитива."""
        if self.primitive_name == "box2d":
            lo, hi = self.primitive_data["lo"], self.primitive_data["hi"]
            n_side = n // 4
            key_left, key_right, key_bottom, key_top = jax.random.split(rng, 4)

            # left (x = lo[0], normal = -x)
            y_l = jax.random.uniform(key_left, (n_side, 1), minval=lo[1], maxval=hi[1])
            pts_l = jnp.hstack([jnp.full((n_side, 1), lo[0]), y_l])
            norms_l = jnp.tile(jnp.array([-1.0, 0.0]), (n_side, 1))

            # right (x = hi[0], normal = +x)
            y_r = jax.random.uniform(key_right, (n_side, 1), minval=lo[1], maxval=hi[1])
            pts_r = jnp.hstack([jnp.full((n_side, 1), hi[0]), y_r])
            norms_r = jnp.tile(jnp.array([1.0, 0.0]), (n_side, 1))

            # bottom (y = lo[1], normal = -y)
            x_b = jax.random.uniform(
                key_bottom, (n_side, 1), minval=lo[0], maxval=hi[0]
            )
            pts_b = jnp.hstack([x_b, jnp.full((n_side, 1), lo[1])])
            norms_b = jnp.tile(jnp.array([0.0, -1.0]), (n_side, 1))

            # top (y = hi[1], normal = +y)
            x_t = jax.random.uniform(key_top, (n_side, 1), minval=lo[0], maxval=hi[0])
            pts_t = jnp.hstack([x_t, jnp.full((n_side, 1), hi[1])])
            norms_t = jnp.tile(jnp.array([0.0, 1.0]), (n_side, 1))

            return {
                "left": (pts_l, norms_l),
                "right": (pts_r, norms_r),
                "bottom": (pts_b, norms_b),
                "top": (pts_t, norms_t),
            }

        elif self.primitive_name == "circle":
            c, r = self.primitive_data["c"], self.primitive_data["r"]
            theta = jax.random.uniform(rng, (n,), minval=0, maxval=2 * jnp.pi)
            pts = jnp.column_stack(
                [c[0] + r * jnp.cos(theta), c[1] + r * jnp.sin(theta)]
            )
            norms = jnp.column_stack([jnp.cos(theta), jnp.sin(theta)])
            return {"outer": (pts, norms)}

        raise ValueError(f"Unknown primitive: {self.primitive_name}")

    def _get_candidate_faces(self, n, rng):
        """Рекурсивно собирает грани детей и объединяет их с учётом операции."""
        if not self.children:
            return self._primitive_boundary(n, rng)

        a, b, op = self.children
        faces_a = a._get_candidate_faces(n // 2, rng)
        faces_b = b._get_candidate_faces(n // 2, rng)

        # Для difference нормали инструмента (b) инвертируются
        if op == "difference":
            faces_b = {k: (pts, -norms) for k, (pts, norms) in faces_b.items()}

        # Объединение словарей
        result = dict(faces_a)
        for k, v in faces_b.items():
            # Уникальные имена (если совпадают - добавляем индекс)
            new_k = k if k not in result else f"{k}_{len(result)}"
            result[new_k] = v
        return result

    def sample_boundary_by_name(self, n_dict, rng):
        """Возвращает точки и нормали для запрошенных граней с фильтрацией."""
        total = sum(n_dict.values())
        candidates = self._get_candidate_faces(total, rng)

        result = {}
        for name, n in n_dict.items():
            if name not in candidates:
                raise KeyError(f"Unknown face {name!r}. Available: {list(candidates)}")
            pts, norms = candidates[name]

            # 1. Точка на итоговой границе (|sdf| < eps)
            mask = jnp.abs(self.sdf_fn(pts)) < 1e-4

            # 2. Для CSG: точка не внутри "соседа" (b) или вычитаемого
            if self.children:
                _a, b, op = self.children
                if op in ("difference", "union"):
                    mask = mask & (b.sdf_fn(pts) >= -1e-4)

            pts_f = pts[mask][:n]
            norms_f = norms[mask][:n]
            result[name] = (pts_f, norms_f)
        return result

    def sample_interior(self, n, rng):
        """Rejection sampling внутри bounds."""
        lo, hi = self.bounds
        batch = n * 10
        pts = jax.random.uniform(rng, (batch, self.dim), minval=lo, maxval=hi)
        mask = self.is_inside(pts)
        return pts[mask][:n]

    # ============================================================ CSG-операторы
    def __add__(self, other):
        return SDFGeometry(
            sdf_fn=lambda x: csg_union(self.sdf_fn(x), other.sdf_fn(x)),
            bounds=(
                jnp.minimum(self.bounds[0], other.bounds[0]),
                jnp.maximum(self.bounds[1], other.bounds[1]),
            ),
            children=(self, other, "union"),
        )

    def __sub__(self, other):
        return SDFGeometry(
            sdf_fn=lambda x: csg_difference(self.sdf_fn(x), other.sdf_fn(x)),
            bounds=self.bounds,
            children=(self, other, "difference"),
        )

    def __and__(self, other):
        return SDFGeometry(
            sdf_fn=lambda x: csg_intersection(self.sdf_fn(x), other.sdf_fn(x)),
            bounds=(
                jnp.maximum(self.bounds[0], other.bounds[0]),
                jnp.minimum(self.bounds[1], other.bounds[1]),
            ),
            children=(self, other, "intersection"),
        )


# ============================================================ Фабрики


def Box2D(x0, x1, y0, y1):
    lo, hi = jnp.array([x0, y0]), jnp.array([x1, y1])
    return SDFGeometry(
        sdf_fn=lambda x: sdf_box(x, lo, hi),
        bounds=(lo, hi),
        primitive_name="box2d",
        primitive_data={"lo": lo, "hi": hi},
    )


def Circle(cx, cy, r):
    c = jnp.array([cx, cy])
    return SDFGeometry(
        sdf_fn=lambda x: sdf_circle(x, c, r),
        bounds=(c - r, c + r),
        primitive_name="circle",
        primitive_data={"c": c, "r": r},
    )
