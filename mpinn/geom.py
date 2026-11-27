"""Модуль геометрий для генерации точек коллокации, граничных условий и интерфейсов.

Поддерживает 1D, 2D и 3D домены. Все классы наследуются от GeometryBase.
"""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ============================================================ Вспомогательные функции


def rotate_points(
    points: jax.Array, angle: float, center: tuple[float, float] = (0.0, 0.0)
) -> jax.Array:
    """Поворачивает массив точек (N, 2) на заданный угол вокруг центра."""
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    cx, cy = center
    x = points[:, 0] - cx
    y = points[:, 1] - cy
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    return jnp.column_stack([x_rot + cx, y_rot + cy])


def rotate_normals(normals: jax.Array, angle: float) -> jax.Array:
    """Поворачивает массив нормалей (N, 2) на заданный угол."""
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    nx = normals[:, 0]
    ny = normals[:, 1]
    return jnp.column_stack([cos_a * nx - sin_a * ny, sin_a * nx + cos_a * ny])


# ============================================================ Визуализация


def plot_domain(
    geometry: "GeometryBase",
    interior_points: jax.Array,
    boundary_points: jax.Array | tuple[jax.Array, jax.Array] | None = None,
    interface_points: jax.Array | None = None,
    title: str = "Визуализация домена",
    **kwargs: Any,
) -> None:
    """
    Визуализирует домен с точками коллокации.

    Args:
        geometry: Объект GeometryBase для определения размерности.
        interior_points: Внутренние точки (N, dim).
        boundary_points: Граничные точки (M, dim) или кортеж (points, normals).
        interface_points: Точки на интерфейсах (K, dim).
        title: Заголовок графика.
        **kwargs: Доп. аргументы (например, show_normals, scale).
    """
    if geometry.dim == 1:
        _plot_1d(interior_points, boundary_points, interface_points, title)
    elif geometry.dim == 2:
        _plot_2d(interior_points, boundary_points, interface_points, title, **kwargs)
    elif geometry.dim == 3:
        _plot_3d(interior_points, boundary_points, interface_points, title, **kwargs)
    else:
        raise ValueError(f"Визуализация для {geometry.dim}D не поддерживается.")


def _plot_1d(interior, boundary=None, interface=None, title="Domain Visualization"):
    """Отрисовка 1D домена."""
    plt.figure(figsize=(10, 2))

    if interior is not None and len(interior) > 0:
        plt.scatter(
            interior[:, 0],
            jnp.zeros_like(interior[:, 0]),
            c="blue",
            s=30,
            alpha=0.5,
            label="Внутренние точки",
        )

    if boundary is not None and len(boundary) > 0:
        boundary_pts = boundary[0] if isinstance(boundary, tuple) else boundary
        plt.scatter(
            boundary_pts[:, 0],
            jnp.zeros_like(boundary_pts[:, 0]),
            c="green",
            s=50,
            alpha=0.7,
            label="Граница",
        )

    if interface is not None and len(interface) > 0:
        plt.scatter(
            interface[:, 0],
            jnp.zeros_like(interface[:, 0]),
            c="red",
            s=50,
            alpha=0.7,
            label="Интерфейс",
        )

    plt.xlabel("x")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def _plot_2d(
    interior,
    boundary=None,
    interface=None,
    title="Domain Visualization",
    show_normals=False,
    scale=0.1,
):
    """Отрисовка 2D домена."""
    plt.figure(figsize=(8, 6))

    if interior is not None and len(interior) > 0:
        plt.scatter(
            interior[:, 0],
            interior[:, 1],
            c="blue",
            s=10,
            alpha=0.5,
            label="Внутренние точки",
        )

    if boundary is not None and len(boundary) > 0:
        if isinstance(boundary, tuple):
            boundary_pts, normals = boundary
            plt.scatter(
                boundary_pts[:, 0],
                boundary_pts[:, 1],
                c="green",
                s=20,
                alpha=0.7,
                label="Граничные точки",
            )

            if show_normals:
                for i in range(len(boundary_pts)):
                    pt = boundary_pts[i]
                    n = normals[i]
                    plt.arrow(
                        pt[0],
                        pt[1],
                        scale * n[0],
                        scale * n[1],
                        head_width=0.02,
                        head_length=0.03,
                        fc="pink",
                        ec="pink",
                    )
        else:
            plt.scatter(
                boundary[:, 0],
                boundary[:, 1],
                c="green",
                s=20,
                alpha=0.7,
                label="Граница",
            )

    if interface is not None and len(interface) > 0:
        plt.scatter(
            interface[:, 0],
            interface[:, 1],
            c="red",
            s=20,
            alpha=0.7,
            label="Интерфейс",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def _plot_3d(
    interior,
    boundary=None,
    interface=None,
    title="Domain Visualization",
    save_path=None,
    show=True,
):
    """Интерактивная 3D визуализация с использованием Plotly (только маркеры, без линий)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Для 3D визуализации установите plotly: pip install plotly")

    fig = go.Figure()

    if interior is not None and len(interior) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=interior[:, 0],
                y=interior[:, 1],
                z=interior[:, 2],
                mode="markers",
                marker={"size": 3, "color": "blue", "opacity": 0.5},
                name="Внутренние точки",
            )
        )

    if boundary is not None and len(boundary) > 0:
        boundary_pts = boundary[0] if isinstance(boundary, tuple) else boundary
        fig.add_trace(
            go.Scatter3d(
                x=boundary_pts[:, 0],
                y=boundary_pts[:, 1],
                z=boundary_pts[:, 2],
                mode="markers",
                marker={"size": 5, "color": "green", "opacity": 0.7},
                name="Граница",
            )
        )

    if interface is not None and len(interface) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=interface[:, 0],
                y=interface[:, 1],
                z=interface[:, 2],
                mode="markers",
                marker={"size": 5, "color": "red", "opacity": 0.7},
                name="Интерфейс",
            )
        )

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z", "aspectmode": "data"
        },
        width=1200,
        height=600,
        autosize=True,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Сохранено: {save_path}")

    if show:
        fig.show(
            config={"responsive": True, "displayModeBar": True, "displaylogo": False}
        )


# ============================================================ Базовый класс


class GeometryBase(ABC):
    """Абстрактный базовый класс для всех геометрий."""

    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def sample_interior(
        self, n_points: int, method: str = "random", rng: jax.Array | None = None
    ) -> jax.Array:
        """Генерирует n_points внутри домена."""

    @abstractmethod
    def get_boundary_names(self) -> list[str]:
        """Возвращает список имен всех граней/поверхностей домена."""

    @abstractmethod
    def sample_boundary_by_name(
        self,
        n_points_dict: dict[str, int],
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> dict[str, tuple[jax.Array, jax.Array]]:
        """
        Генерирует точки и нормали для указанных именованных граней.

        Returns:
            Словарь вида {имя_грани: (points, normals)}.
        """

    @abstractmethod
    def sample_interface(
        self,
        other_geom: "GeometryBase",
        n_points: int,
        self_interface_name: str,
        other_interface_name: str,
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Генерирует точки на интерфейсе между текущей и другой геометрией."""

    @property
    @abstractmethod
    def bounds(self) -> tuple[jax.Array, jax.Array]:
        """Возвращает (min_coords, max_coords) ограничивающего бокса."""


# ============================================================ 1D: Интервал


class Interval(GeometryBase):
    """1D интервал [x0, x1]."""

    def __init__(self, x0: float, x1: float):
        super().__init__(dim=1)
        self.x0 = float(x0)
        self.x1 = float(x1)

    def sample_interior(
        self, n_points: int, method: str = "random", rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if method == "random":
            return jax.random.uniform(
                rng, shape=(n_points, 1), minval=self.x0, maxval=self.x1
            )
        if method == "uniform":
            return jnp.linspace(self.x0, self.x1, n_points).reshape(-1, 1)
        raise ValueError(f"Неизвестный метод: {method}")

    def get_boundary_names(self) -> list[str]:
        return ["left", "right"]

    def _generate_boundary_points(
        self, name: str, n: int, rng: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        if name == "left":
            pts = jnp.full((n, 1), self.x0)
            norms = jnp.full((n, 1), -1.0)
        elif name == "right":
            pts = jnp.full((n, 1), self.x1)
            norms = jnp.full((n, 1), 1.0)
        else:
            raise ValueError(f"Неизвестная грань '{name}' для Interval")
        return pts, norms

    def sample_boundary_by_name(
        self,
        n_points_dict: dict[str, int],
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> dict[str, tuple[jax.Array, jax.Array]]:
        result = {}
        for name, n in n_points_dict.items():
            if n > 0:
                result[name] = self._generate_boundary_points(name, n)
        return result

    def sample_interface(
        self,
        other_geom: GeometryBase,
        n_points: int,
        self_interface_name: str,
        other_interface_name: str,
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if not isinstance(other_geom, Interval):
            raise TypeError("other_geom должен быть Interval")
        return self._generate_boundary_points(self_interface_name, n_points)

    @property
    def bounds(self) -> tuple[jax.Array, jax.Array]:
        return jnp.array([self.x0]), jnp.array([self.x1])


# ============================================================ 2D: Прямоугольник


class Box2D(GeometryBase):
    """2D прямоугольник."""

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        super().__init__(dim=2)
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)

    def sample_interior(
        self, n_points: int, method: str = "random", rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == "random":
            keys = jax.random.split(rng, 2)
            x = jax.random.uniform(
                keys[0], (n_points, 1), minval=self.x_min, maxval=self.x_max
            )
            y = jax.random.uniform(
                keys[1], (n_points, 1), minval=self.y_min, maxval=self.y_max
            )
            return jnp.hstack([x, y])
        if method == "uniform":
            n_per_side = int(jnp.sqrt(n_points))
            x = jnp.linspace(self.x_min, self.x_max, n_per_side)
            y = jnp.linspace(self.y_min, self.y_max, n_per_side)
            xx, yy = jnp.meshgrid(x, y, indexing="ij")
            return jnp.stack([xx.ravel(), yy.ravel()], axis=-1)[:n_points]
        raise ValueError(f"Неизвестный метод: {method}")

    def get_boundary_names(self) -> list[str]:
        return ["left", "right", "bottom", "top"]

    def _generate_boundary_points(
        self, name: str, n: int, method: str = "random", rng: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        t = (
            jax.random.uniform(rng, (n, 1), minval=0.0, maxval=1.0)
            if method == "random"
            else jnp.linspace(0, 1, n).reshape(-1, 1)
        )

        if name == "left":
            start, end, normal = (
                jnp.array([self.x_min, self.y_min]),
                jnp.array([self.x_min, self.y_max]),
                jnp.array([-1.0, 0.0]),
            )
        elif name == "right":
            start, end, normal = (
                jnp.array([self.x_max, self.y_min]),
                jnp.array([self.x_max, self.y_max]),
                jnp.array([1.0, 0.0]),
            )
        elif name == "bottom":
            start, end, normal = (
                jnp.array([self.x_min, self.y_min]),
                jnp.array([self.x_max, self.y_min]),
                jnp.array([0.0, -1.0]),
            )
        elif name == "top":
            start, end, normal = (
                jnp.array([self.x_min, self.y_max]),
                jnp.array([self.x_max, self.y_max]),
                jnp.array([0.0, 1.0]),
            )
        else:
            raise ValueError(f"Неизвестная грань '{name}' для Box2D")

        pts = start + t * (end - start)
        return pts, jnp.tile(normal, (n, 1))

    def sample_boundary_by_name(
        self,
        n_points_dict: dict[str, int],
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> dict[str, tuple[jax.Array, jax.Array]]:
        result = {}
        keys = jax.random.split(rng, 4) if rng is not None else [None] * 4
        for i, name in enumerate(self.get_boundary_names()):
            n = n_points_dict.get(name, 0)
            if n > 0:
                result[name] = self._generate_boundary_points(name, n, method, keys[i])
        return result

    def sample_interface(
        self,
        other_geom: GeometryBase,
        n_points: int,
        self_interface_name: str,
        other_interface_name: str,
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if not isinstance(other_geom, Box2D):
            raise TypeError("other_geom должен быть Box2D")
        return self._generate_boundary_points(
            self_interface_name, n_points, method, rng
        )

    @property
    def bounds(self) -> tuple[jax.Array, jax.Array]:
        return jnp.array([self.x_min, self.y_min]), jnp.array([self.x_max, self.y_max])


# ============================================================ 3D: Параллелепипед


class Box3D(GeometryBase):
    """3D параллелепипед."""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
    ):
        super().__init__(dim=3)
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.z_min, self.z_max = float(z_min), float(z_max)

    def sample_interior(
        self, n_points: int, method: str = "random", rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if method == "random":
            keys = jax.random.split(rng, 3)
            x = jax.random.uniform(
                keys[0], (n_points, 1), minval=self.x_min, maxval=self.x_max
            )
            y = jax.random.uniform(
                keys[1], (n_points, 1), minval=self.y_min, maxval=self.y_max
            )
            z = jax.random.uniform(
                keys[2], (n_points, 1), minval=self.z_min, maxval=self.z_max
            )
            return jnp.hstack([x, y, z])
        raise ValueError(f"Неизвестный метод: {method}")

    def get_boundary_names(self) -> list[str]:
        return ["left", "right", "front", "back", "bottom", "top"]

    def _generate_boundary_points(
        self, name: str, n: int, method: str = "random", rng: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # ВАЖНО: разбиваем ключ на два, чтобы координаты не были коррелированы!
        k1, k2 = jax.random.split(rng)

        if name == "left":
            pts = jnp.hstack(
                [
                    jnp.full((n, 1), self.x_min),
                    jax.random.uniform(
                        k1, (n, 1), minval=self.y_min, maxval=self.y_max
                    ),
                    jax.random.uniform(
                        k2, (n, 1), minval=self.z_min, maxval=self.z_max
                    ),
                ]
            )
            normal = jnp.array([-1.0, 0.0, 0.0])
        elif name == "right":
            pts = jnp.hstack(
                [
                    jnp.full((n, 1), self.x_max),
                    jax.random.uniform(
                        k1, (n, 1), minval=self.y_min, maxval=self.y_max
                    ),
                    jax.random.uniform(
                        k2, (n, 1), minval=self.z_min, maxval=self.z_max
                    ),
                ]
            )
            normal = jnp.array([1.0, 0.0, 0.0])
        elif name == "front":
            pts = jnp.hstack(
                [
                    jax.random.uniform(
                        k1, (n, 1), minval=self.x_min, maxval=self.x_max
                    ),
                    jnp.full((n, 1), self.y_min),
                    jax.random.uniform(
                        k2, (n, 1), minval=self.z_min, maxval=self.z_max
                    ),
                ]
            )
            normal = jnp.array([0.0, -1.0, 0.0])
        elif name == "back":
            pts = jnp.hstack(
                [
                    jax.random.uniform(
                        k1, (n, 1), minval=self.x_min, maxval=self.x_max
                    ),
                    jnp.full((n, 1), self.y_max),
                    jax.random.uniform(
                        k2, (n, 1), minval=self.z_min, maxval=self.z_max
                    ),
                ]
            )
            normal = jnp.array([0.0, 1.0, 0.0])
        elif name == "bottom":
            pts = jnp.hstack(
                [
                    jax.random.uniform(
                        k1, (n, 1), minval=self.x_min, maxval=self.x_max
                    ),
                    jax.random.uniform(
                        k2, (n, 1), minval=self.y_min, maxval=self.y_max
                    ),
                    jnp.full((n, 1), self.z_min),
                ]
            )
            normal = jnp.array([0.0, 0.0, -1.0])
        elif name == "top":
            pts = jnp.hstack(
                [
                    jax.random.uniform(
                        k1, (n, 1), minval=self.x_min, maxval=self.x_max
                    ),
                    jax.random.uniform(
                        k2, (n, 1), minval=self.y_min, maxval=self.y_max
                    ),
                    jnp.full((n, 1), self.z_max),
                ]
            )
            normal = jnp.array([0.0, 0.0, 1.0])
        else:
            raise ValueError(f"Неизвестная грань '{name}' для Box3D")

        return pts, jnp.tile(normal.reshape(1, 3), (n, 1))

    def sample_boundary_by_name(
        self,
        n_points_dict: dict[str, int],
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> dict[str, tuple[jax.Array, jax.Array]]:
        result = {}
        keys = jax.random.split(rng, 6) if rng is not None else [None] * 6
        for i, name in enumerate(self.get_boundary_names()):
            n = n_points_dict.get(name, 0)
            if n > 0:
                result[name] = self._generate_boundary_points(name, n, method, keys[i])
        return result

    def sample_interface(
        self,
        other_geom: GeometryBase,
        n_points: int,
        self_interface_name: str,
        other_interface_name: str,
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if not isinstance(other_geom, Box3D):
            raise TypeError("other_geom должен быть Box3D")
        return self._generate_boundary_points(
            self_interface_name, n_points, method, rng
        )

    @property
    def bounds(self) -> tuple[jax.Array, jax.Array]:
        return jnp.array([self.x_min, self.y_min, self.z_min]), jnp.array(
            [self.x_max, self.y_max, self.z_max]
        )


# ============================================================ 3D: Сфера


class Sphere(GeometryBase):
    """3D сфера (сплошная)."""

    def __init__(
        self, cx: float = 0.0, cy: float = 0.0, cz: float = 0.0, radius: float = 1.0
    ):
        super().__init__(dim=3)
        self.cx, self.cy, self.cz = float(cx), float(cy), float(cz)
        self.radius = float(radius)

    def sample_interior(
        self, n_points: int, method: str = "random", rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if method == "random":
            keys = jax.random.split(rng, 3)
            r = self.radius * jnp.cbrt(jax.random.uniform(keys[0], (n_points, 1)))
            theta = 2 * jnp.pi * jax.random.uniform(keys[1], (n_points, 1))
            phi = jnp.arccos(2 * jax.random.uniform(keys[2], (n_points, 1)) - 1)

            x = self.cx + r * jnp.sin(phi) * jnp.cos(theta)
            y = self.cy + r * jnp.sin(phi) * jnp.sin(theta)
            z = self.cz + r * jnp.cos(phi)
            return jnp.hstack([x, y, z])
        raise ValueError(f"Неизвестный метод: {method}")

    def get_boundary_names(self) -> list[str]:
        return ["outer"]

    def _generate_boundary_points(
        self, name: str, n: int, method: str = "random", rng: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        if name != "outer":
            raise ValueError(f"Sphere имеет только грань 'outer', получено '{name}'")
        if rng is None:
            rng = jax.random.PRNGKey(0)

        keys = jax.random.split(rng, 2)
        theta = 2 * jnp.pi * jax.random.uniform(keys[0], (n, 1))
        phi = jnp.arccos(2 * jax.random.uniform(keys[1], (n, 1)) - 1)

        x = self.cx + self.radius * jnp.sin(phi) * jnp.cos(theta)
        y = self.cy + self.radius * jnp.sin(phi) * jnp.sin(theta)
        z = self.cz + self.radius * jnp.cos(phi)
        pts = jnp.hstack([x, y, z])

        normals = jnp.hstack(
            [jnp.sin(phi) * jnp.cos(theta), jnp.sin(phi) * jnp.sin(theta), jnp.cos(phi)]
        )
        return pts, normals

    def sample_boundary_by_name(
        self,
        n_points_dict: dict[str, int],
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> dict[str, tuple[jax.Array, jax.Array]]:
        result = {}
        for name, n in n_points_dict.items():
            if n > 0:
                result[name] = self._generate_boundary_points(name, n, method, rng)
        return result

    def sample_interface(
        self,
        other_geom: GeometryBase,
        n_points: int,
        self_interface_name: str,
        other_interface_name: str,
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if not isinstance(other_geom, Sphere):
            raise TypeError("other_geom должен быть Sphere")
        if self_interface_name != "outer":
            raise ValueError("Интерфейс Sphere поддерживается только для грани 'outer'")
        return self._generate_boundary_points("outer", n_points, method, rng)

    @property
    def bounds(self) -> tuple[jax.Array, jax.Array]:
        R = self.radius
        return jnp.array([self.cx - R, self.cy - R, self.cz - R]), jnp.array(
            [self.cx + R, self.cy + R, self.cz + R]
        )


# ============================================================ 2D: Кольцо


class Annulus2D(GeometryBase):
    """2D кольцо (полый круг)."""

    def __init__(
        self, center: tuple[float, float], R_outer: float, R_inner: float = 0.0
    ):
        if R_outer <= 0 or R_inner < 0 or R_inner >= R_outer:
            raise ValueError("Некорректные радиусы кольца.")
        
        super().__init__(dim=2)
        self.center = jnp.array(center, dtype=jnp.float32)
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)
        self.is_hollow = R_inner > 0

    def sample_interior(
        self, n_points: int, method: str = "random", rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        key1, key2 = jax.random.split(rng, 2)
        u = jax.random.uniform(key1, (n_points,))
        theta = jax.random.uniform(key2, (n_points,), minval=0, maxval=2 * jnp.pi)

        r = (
            jnp.sqrt(self.R_inner**2 + u * (self.R_outer**2 - self.R_inner**2))
            if self.is_hollow
            else jnp.sqrt(u) * self.R_outer
        )

        x = self.center[0] + r * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(theta)
        points = jnp.column_stack([x, y])
        return jax.random.permutation(rng, points, axis=0)

    def get_boundary_names(self) -> list[str]:
        return ["outer", "inner"] if self.is_hollow else ["outer"]

    def _generate_boundary_points(
        self, name: str, n: int, method: str = "random", rng: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if name == "outer":
            R, sign = self.R_outer, 1.0
        elif name == "inner":
            if not self.is_hollow:
                raise ValueError("Внутренняя граница недоступна (сплошной диск)")
            R, sign = self.R_inner, -1.0
        else:
            raise ValueError(f"Неизвестная грань '{name}' для Annulus2D")

        theta = jax.random.uniform(rng, (n,), minval=0, maxval=2 * jnp.pi)
        x = self.center[0] + R * jnp.cos(theta)
        y = self.center[1] + R * jnp.sin(theta)
        pts = jnp.column_stack([x, y])
        normals = sign * jnp.column_stack([jnp.cos(theta), jnp.sin(theta)])
        return pts, normals

    def sample_boundary_by_name(
        self,
        n_points_dict: dict[str, int],
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> dict[str, tuple[jax.Array, jax.Array]]:
        result = {}
        names = self.get_boundary_names()
        keys = jax.random.split(rng, len(names)) if rng is not None else [None] * len(names)
        for i, name in enumerate(names):
            n = n_points_dict.get(name, 0)
            if n > 0:
                result[name] = self._generate_boundary_points(name, n, method, keys[i])
        return result


    def is_inside(self, points: jax.Array) -> jax.Array:
        """Проверяет, лежат ли точки внутри кольца."""
        r = jnp.linalg.norm(points - self.center, axis=1)
        return (r <= self.R_outer) & (r >= self.R_inner)

    def sample_interface(
        self,
        other_geom: GeometryBase,
        n_points: int,
        self_interface_name: str,
        other_interface_name: str,
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if not isinstance(other_geom, Annulus2D):
            raise TypeError("other_geom должен быть Annulus2D")
        return self._generate_boundary_points(
            self_interface_name, n_points, method, rng
        )

    @property
    def bounds(self) -> tuple[jax.Array, jax.Array]:
        return self.center - self.R_outer, self.center + self.R_outer


# ============================================================ 3D: Полый цилиндр


class HollowCylinder3D(GeometryBase):
    """3D полый цилиндр (труба)."""

    def __init__(
        self,
        center_base: tuple[float, float, float],
        radius_outer: float,
        radius_inner: float,
        height: float,
        axis: tuple[float, float, float] = (0, 0, 1),
    ):
        if (
            radius_outer <= 0
            or radius_inner <= 0
            or radius_inner >= radius_outer
            or height <= 0
        ):
            raise ValueError("Некорректные параметры цилиндра.")
        super().__init__(dim=3)
        self.center_base = jnp.array(center_base, dtype=jnp.float64)
        self.radius_outer = float(radius_outer)
        self.radius_inner = float(radius_inner)
        self.height = float(height)
        self.axis = jnp.array(axis, dtype=jnp.float64)
        self.axis = self.axis / jnp.linalg.norm(self.axis)

    def _cylinder_basis(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        ez = self.axis
        er_ref = (
            jnp.cross(ez, jnp.array([0.0, 0.0, 1.0]))
            if jnp.abs(ez[2]) < 0.9
            else jnp.cross(ez, jnp.array([1.0, 0.0, 0.0]))
        )
        er_ref = er_ref / jnp.linalg.norm(er_ref)
        return ez, er_ref, jnp.cross(ez, er_ref)

    def _cylindrical_to_cartesian(
        self, r: jax.Array, theta: jax.Array, z: jax.Array
    ) -> jax.Array:
        ez, er_ref, etheta_ref = self._cylinder_basis()
        x_local = (
            (r * jnp.cos(theta))[:, None] * er_ref
            + (r * jnp.sin(theta))[:, None] * etheta_ref
            + z[:, None] * ez
        )
        return self.center_base + x_local

    def sample_interior(
        self, n_points: int, method: str = "random", rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 3)
        u = jax.random.uniform(keys[0], (n_points,))
        theta = jax.random.uniform(keys[1], (n_points,), minval=0.0, maxval=2 * jnp.pi)
        z = jax.random.uniform(keys[2], (n_points,), minval=0.0, maxval=self.height)
        r = jnp.sqrt(
            self.radius_inner**2 + u * (self.radius_outer**2 - self.radius_inner**2)
        )
        return self._cylindrical_to_cartesian(r, theta, z)

    def get_boundary_names(self) -> list[str]:
        return ["outer", "inner", "bottom", "top"]

    def _generate_boundary_points(
        self, name: str, n: int, method: str = "random", rng: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # ВАЖНО: разбиваем ключ!
        k1, k2 = jax.random.split(rng)
        _ez, er_ref, etheta_ref = self._cylinder_basis()

        if name in ("outer", "inner"):
            theta = jax.random.uniform(k1, (n,), minval=0.0, maxval=2 * jnp.pi)
            z = jax.random.uniform(k2, (n,), minval=0.0, maxval=self.height)
            r = jnp.full(n, self.radius_outer if name == "outer" else self.radius_inner)
            sign = 1.0 if name == "outer" else -1.0
            pts = self._cylindrical_to_cartesian(r, theta, z)
            norms = sign * (
                jnp.cos(theta)[:, None] * er_ref + jnp.sin(theta)[:, None] * etheta_ref
            )
        elif name in ("bottom", "top"):
            u = jax.random.uniform(k1, (n,))
            theta = jax.random.uniform(k2, (n,), minval=0.0, maxval=2 * jnp.pi)
            r = jnp.sqrt(
                self.radius_inner**2 + u * (self.radius_outer**2 - self.radius_inner**2)
            )
            z = jnp.zeros(n) if name == "bottom" else jnp.full(n, self.height)
            sign = -1.0 if name == "bottom" else 1.0
            pts = self._cylindrical_to_cartesian(r, theta, z)
            norms = jnp.tile(sign * self.axis, (n, 1))
        else:
            raise ValueError(f"Неизвестная грань '{name}' для HollowCylinder3D")

        return pts, norms

    def sample_boundary_by_name(
        self,
        n_points_dict: dict[str, int],
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> dict[str, tuple[jax.Array, jax.Array]]:
        result = {}
        keys = jax.random.split(rng, 4) if rng is not None else [None] * 4
        for i, name in enumerate(self.get_boundary_names()):
            n = n_points_dict.get(name, 0)
            if n > 0:
                result[name] = self._generate_boundary_points(name, n, method, keys[i])
        return result

    def is_inside(self, points: jax.Array) -> jax.Array:
        vec = points - self.center_base
        z_coord = jnp.dot(vec, self.axis)
        r = jnp.linalg.norm(vec - z_coord[:, None] * self.axis, axis=1)
        return (
            (r <= self.radius_outer)
            & (r >= self.radius_inner)
            & (z_coord >= 0)
            & (z_coord <= self.height)
        )

    def sample_interface(
        self,
        other_geom: GeometryBase,
        n_points: int,
        self_interface_name: str,
        other_interface_name: str,
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if not isinstance(other_geom, HollowCylinder3D):
            raise TypeError("other_geom должен быть HollowCylinder3D")
        return self._generate_boundary_points(
            self_interface_name, n_points, method, rng
        )

    @property
    def bounds(self) -> tuple[jax.Array, jax.Array]:
        R, H = self.radius_outer, self.height
        return self.center_base - jnp.array([R, R, 0.0]), self.center_base + jnp.array(
            [R, R, H]
        )


# ============================================================ 3D: Полый шар


class HollowSphere3D(GeometryBase):
    """3D полая сфера."""

    def __init__(
        self, center: tuple[float, float, float], R_outer: float, R_inner: float
    ):
        if R_outer <= 0 or R_inner <= 0 or R_inner >= R_outer:
            raise ValueError("Некорректные радиусы сферы.")
        super().__init__(dim=3)
        self.center = jnp.array(center, dtype=jnp.float64)
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)

    def sample_interior(
        self, n_points: int, method: str = "random", rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 3)
        u, v, w = [jax.random.uniform(k, (n_points,)) for k in keys]

        theta = 2 * jnp.pi * u
        phi = jnp.arccos(2 * v - 1)
        r = (self.R_inner**3 + w * (self.R_outer**3 - self.R_inner**3)) ** (1 / 3)

        x = self.center[0] + r * jnp.sin(phi) * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(phi) * jnp.sin(theta)
        z = self.center[2] + r * jnp.cos(phi)
        return jnp.column_stack([x, y, z])

    def get_boundary_names(self) -> list[str]:
        return ["outer", "inner"]

    def _generate_boundary_points(
        self, name: str, n: int, method: str = "random", rng: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        R = self.R_outer if name == "outer" else self.R_inner
        sign = 1.0 if name == "outer" else -1.0

        keys = jax.random.split(rng, 2)
        theta = 2 * jnp.pi * jax.random.uniform(keys[0], (n,))
        phi = jnp.arccos(2 * jax.random.uniform(keys[1], (n,)) - 1)

        sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)
        sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)

        pts = jnp.column_stack(
            [
                self.center[0] + R * sin_phi * cos_t,
                self.center[1] + R * sin_phi * sin_t,
                self.center[2] + R * cos_phi,
            ]
        )
        norms = sign * jnp.column_stack([sin_phi * cos_t, sin_phi * sin_t, cos_phi])
        return pts, norms

    def sample_boundary_by_name(
        self,
        n_points_dict: dict[str, int],
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> dict[str, tuple[jax.Array, jax.Array]]:
        result = {}
        keys = jax.random.split(rng, 2) if rng is not None else [None] * 2
        for i, name in enumerate(self.get_boundary_names()):
            n = n_points_dict.get(name, 0)
            if n > 0:
                result[name] = self._generate_boundary_points(name, n, method, keys[i])
        return result

    def is_inside(self, points: jax.Array) -> jax.Array:
        r = jnp.linalg.norm(points - self.center, axis=1)
        return (r <= self.R_outer) & (r >= self.R_inner)

    def sample_interface(
        self,
        other_geom: GeometryBase,
        n_points: int,
        self_interface_name: str,
        other_interface_name: str,
        method: str = "random",
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if not isinstance(other_geom, HollowSphere3D):
            raise TypeError("other_geom должен быть HollowSphere3D")
        return self._generate_boundary_points(
            self_interface_name, n_points, method, rng
        )

    @property
    def bounds(self) -> tuple[jax.Array, jax.Array]:
        return self.center - self.R_outer, self.center + self.R_outer
