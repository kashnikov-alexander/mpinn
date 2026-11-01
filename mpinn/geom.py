from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any

def rotate_points(points, angle, center=(0.0, 0.0)):
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    cx, cy = center
    x = points[:, 0] - cx
    y = points[:, 1] - cy
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    return jnp.column_stack([x_rot + cx, y_rot + cy])
    
def rotate_normals(normals, angle):
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    nx = normals[:, 0]
    ny = normals[:, 1]
    return jnp.column_stack([cos_a * nx - sin_a * ny,
                             sin_a * nx + cos_a * ny])


def plot_domain(geometry, interior_points, boundary_points=None, interface_points=None, title="Визуализация домена", **kwargs):
    """
    Визуализирует домен с точками коллокации.

    Args:
        geometry: объект GeometryBase для определения размерности
        interior_points: внутренние точки (N, dim)
        boundary_points: граничные точки (M, dim), опционально
        interface_points: точки на интерфейсах доменов (K, dim), опционально
        title: заголовок графика
        **kwargs: дополнительные аргументы для специфичных типов геометрии (например, show_normals, normal_scale)
    """
    if geometry.dim == 1:
        _plot_1d(interior_points, boundary_points, interface_points, title)
    elif geometry.dim == 2:
        _plot_2d(interior_points, boundary_points, interface_points, title, **kwargs)
    elif geometry.dim == 3:
        _plot_3d(interior_points, boundary_points, interface_points, title, **kwargs)
    else:
        raise ValueError(f"Visualization not supported for {geometry.dim}D")


def _plot_1d(interior, boundary=None, interface=None, title="Domain Visualization"):
    """Отрисовка 1D домена"""
    plt.figure(figsize=(10, 2))

    # Внутренние точки - синие
    if interior is not None and len(interior) > 0:
        plt.scatter(
            interior[:, 0],
            jnp.zeros_like(interior[:, 0]),
            c="blue",
            s=30,
            alpha=0.5,
            label="Внутренние точки",
        )

    # Граничные точки - зеленые
    if boundary is not None and len(boundary) > 0:
        if isinstance(boundary, tuple):
            boundary_pts, _ = boundary
        else:
            boundary_pts = boundary
        plt.scatter(
            boundary_pts[:, 0],
            jnp.zeros_like(boundary_pts[:, 0]),
            c="green",
            s=50,
            alpha=0.7,
            label="Граница",
        )

    # Точки на интерфейсах - красные
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


def _plot_2d(interior, boundary=None, interface=None, title="Domain Visualization", show_normals=False, scale=0.1):
    """Отрисовка 2D домена"""
    plt.figure(figsize=(8, 6))

    # Внутренние точки - синие
    if interior is not None and len(interior) > 0:
        plt.scatter(
            interior[:, 0],
            interior[:, 1],
            c="blue",
            s=10,
            alpha=0.5,
            label="Внутренние точки",
        )

    # Граничные точки - зеленые
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

    # Точки на интерфейсах - красные
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


def _plot_3d(interior, boundary=None, interface=None, title="Domain Visualization", save_path=None):
    """
    Интерактивная 3D визуализация с использованием Plotly.
    
    Args:
        interior: внутренние точки (N, 3)
        boundary: граничные точки (M, 3) или кортеж (points, normals)
        interface: точки на интерфейсах (K, 3)
        title: заголовок
        save_path: путь для сохранения HTML
    """
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    # Внутренние точки
    if interior is not None and len(interior) > 0:
        fig.add_trace(go.Scatter3d(
            x=interior[:, 0],
            y=interior[:, 1],
            z=interior[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.5),
            name='Внутренние точки'
        ))
    
    # Граничные точки
    if boundary is not None and len(boundary) > 0:
        if isinstance(boundary, tuple):
            boundary_pts, normals = boundary
        else:
            boundary_pts = boundary
            normals = None
        
        fig.add_trace(go.Scatter3d(
            x=boundary_pts[:, 0],
            y=boundary_pts[:, 1],
            z=boundary_pts[:, 2],
            mode='markers',
            marker=dict(size=5, color='green', opacity=0.7),
            name='Граница'
        ))
    
    # Интерфейсные точки
    if interface is not None and len(interface) > 0:
        fig.add_trace(go.Scatter3d(
            x=interface[:, 0],
            y=interface[:, 1],
            z=interface[:, 2],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.7),
            name='Интерфейс'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='data'
        ),
        width=1200,
        height=600,
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Сохранено: {save_path}")
    
    fig.show(
        config={
        'responsive': True,          # адаптивный размер
        'displayModeBar': True,      # панель инструментов
        'displaylogo': False,        # скрыть логотип Plotly
        }
    )

class GeometryBase(ABC):
    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def sample_interior(self, n_points, method="random", rng=None):
        pass

    @abstractmethod
    def sample_boundary(self, n_points=None, method="random", rng=None):
        pass

    # ===== НОВЫЕ МЕТОДЫ =====
    @abstractmethod
    def get_boundary_names(self) -> List[str]:
        """Возвращает список имён всех граней."""
        pass

    @abstractmethod
    def sample_boundary_by_name(self, n_points_dict: Dict[str, int], method="random", rng=None) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Генерирует точки и нормали для указанных граней.
        Args:
            n_points_dict: {имя_грани: количество_точек}
        Returns:
            {имя_грани: (points, normals)}
        """
        pass

    @abstractmethod
    def sample_interface(self, other_geom: 'GeometryBase', n_points: int,
                         self_interface_name: str, other_interface_name: str,
                         method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Генерирует точки на общей границе (интерфейсе) между текущей геометрией и другой.
        Args:
            other_geom: соседняя геометрия
            n_points: количество точек
            self_interface_name: имя грани в текущей геометрии
            other_interface_name: имя грани в другой геометрии
        Returns:
            (points, normals_self, normals_other)
        """
        pass

    @property
    @abstractmethod
    def bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Возвращает (min_coords, max_coords) как массивы JAX."""
        pass


class Interval(GeometryBase):
    def __init__(self, x0, x1):
        super().__init__(dim=1)
        self.x0 = float(x0)
        self.x1 = float(x1)

    def sample_interior(self, n_points, method="random", rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if method == "random":
            return jax.random.uniform(rng, shape=(n_points, 1), minval=self.x0, maxval=self.x1)
        if method == "uniform":
            return jnp.linspace(self.x0, self.x1, n_points).reshape(-1, 1)
        raise ValueError(f"Unknown method: {method}")

    def sample_boundary(self, n_points=None, method="random", rng=None):
        # Старый метод оставлен для совместимости
        points = jnp.array([[self.x0], [self.x1]])
        normals = jnp.array([[-1.0], [1.0]])
        return points, normals

    def generate_collocation(self, n_interior=100, method="random", rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)

        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method, keys[0])
        boundary, _ = self.sample_boundary()

        return jnp.vstack([interior, boundary])
        
        # ===== НОВЫЕ МЕТОДЫ =====
    def get_boundary_names(self) -> List[str]:
        return ['left', 'right']

    def _generate_boundary_points(self, name: str, n: int, method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if name == 'left':
            pts = jnp.full((n, 1), self.x0)
            norms = jnp.full((n, 1), -1.0)
        elif name == 'right':
            pts = jnp.full((n, 1), self.x1)
            norms = jnp.full((n, 1), 1.0)
        else:
            raise ValueError(f"Unknown boundary name '{name}' for Interval")
        return pts, norms

    def sample_boundary_by_name(self, n_points_dict: Dict[str, int], method="random", rng=None) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        result = {}
        for name, n in n_points_dict.items():
            if n > 0:
                pts, norms = self._generate_boundary_points(name, n, method, rng)
                result[name] = (pts, norms)
        return result

    def sample_interface(self, other_geom: 'GeometryBase', n_points: int,
                         self_interface_name: str, other_interface_name: str,
                         method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not isinstance(other_geom, Interval):
            raise TypeError("other_geom must be Interval")
        pts, normals_self = self._generate_boundary_points(self_interface_name, n_points, method, rng)
        return pts, normals_self

    @property
    def bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.array([self.x0]), jnp.array([self.x1])


class Box2D(GeometryBase):
    def __init__(self, x_min, x_max, y_min, y_max):
        super().__init__(dim=2)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def sample_interior(self, n_points, method="random", rng=None):
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
            points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
            return points[:n_points]
        raise ValueError(f"Неизвестный метод: {method}")

    def sample_boundary(self, n_points, method="random", rng=None, return_edges=False):
        """
        Генерирует точки на границе. Точки распределяются по сторонам пропорционально длинам.
        n_points должно быть > 0.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        def _points_on_segment(start, end, n, rng_sub):
            if n <= 0:
                return jnp.empty((0, 2))
            if method == "random":
                t = jax.random.uniform(rng_sub, (n, 1), minval=0.0, maxval=1.0)
            else:
                t = jnp.linspace(0, 1, n).reshape(-1, 1)
            return start + t * (end - start)

        sides = {
            'left':   (jnp.array([self.x_min, self.y_min]), jnp.array([self.x_min, self.y_max]), jnp.array([-1.0, 0.0])),
            'right':  (jnp.array([self.x_max, self.y_min]), jnp.array([self.x_max, self.y_max]), jnp.array([ 1.0, 0.0])),
            'bottom': (jnp.array([self.x_min, self.y_min]), jnp.array([self.x_max, self.y_min]), jnp.array([ 0.0,-1.0])),
            'top':    (jnp.array([self.x_min, self.y_max]), jnp.array([self.x_max, self.y_max]), jnp.array([ 0.0, 1.0])),
        }

        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        perim = 2 * (dx + dy)
        lengths = jnp.array([dy, dy, dx, dx])
        fractions = lengths / perim
        n_side = jnp.round(fractions * n_points).astype(int)
        diff = n_points - jnp.sum(n_side)
        if diff != 0:
            idx = jnp.argmax(lengths)
            n_side = n_side.at[idx].add(diff)

        keys = jax.random.split(rng, 4) if method == "random" else [None] * 4
        result = {}
        side_names = ['left', 'right', 'bottom', 'top']
        for name, (start, end, normal), n, k in zip(side_names, sides.values(), n_side, keys):
            pts = _points_on_segment(start, end, int(n), k)
            norms = jnp.tile(normal.reshape(1, 2), (int(n), 1)) if n > 0 else jnp.empty((0, 2))
            result[name] = (pts, norms)

        if return_edges:
            return result
        else:
            all_pts = jnp.vstack([v[0] for v in result.values()])
            all_norms = jnp.vstack([v[1] for v in result.values()])
            return all_pts, all_norms

    def generate_collocation(
        self,
        n_interior,
        n_boundary,
        method_interior="random",
        method_boundary="random",
        rng=None,
    ):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary, _ = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])
        
    # ===== НОВЫЕ МЕТОДЫ =====
    def get_boundary_names(self) -> List[str]:
        return ['left', 'right', 'bottom', 'top']

    def _generate_boundary_points(self, name: str, n: int, method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if method == "random":
            t = jax.random.uniform(rng, (n, 1), minval=0.0, maxval=1.0)
        else:
            t = jnp.linspace(0, 1, n).reshape(-1, 1)

        if name == 'left':
            start = jnp.array([self.x_min, self.y_min])
            end = jnp.array([self.x_min, self.y_max])
            normal = jnp.array([-1.0, 0.0])
        elif name == 'right':
            start = jnp.array([self.x_max, self.y_min])
            end = jnp.array([self.x_max, self.y_max])
            normal = jnp.array([1.0, 0.0])
        elif name == 'bottom':
            start = jnp.array([self.x_min, self.y_min])
            end = jnp.array([self.x_max, self.y_min])
            normal = jnp.array([0.0, -1.0])
        elif name == 'top':
            start = jnp.array([self.x_min, self.y_max])
            end = jnp.array([self.x_max, self.y_max])
            normal = jnp.array([0.0, 1.0])
        else:
            raise ValueError(f"Unknown boundary name '{name}' for Box2D")
        pts = start + t * (end - start)
        norms = jnp.tile(normal.reshape(1, 2), (n, 1))
        return pts, norms

    def sample_boundary_by_name(self, n_points_dict: Dict[str, int], method="random", rng=None) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        result = {}
        keys = jax.random.split(rng, 4) if method == "random" and rng is not None else [None] * 4
        for i, name in enumerate(self.get_boundary_names()):
            n = n_points_dict.get(name, 0)
            if n > 0:
                pts, norms = self._generate_boundary_points(name, n, method, keys[i])
                result[name] = (pts, norms)
        return result

    def sample_interface(self, other_geom: 'GeometryBase', n_points: int,
                         self_interface_name: str, other_interface_name: str,
                         method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not isinstance(other_geom, Box2D):
            raise TypeError("other_geom must be Box2D")
        pts, normals_self = self._generate_boundary_points(self_interface_name, n_points, method, rng)
        return pts, normals_self

    @property
    def bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.array([self.x_min, self.y_min]), jnp.array([self.x_max, self.y_max])


class Box3D(GeometryBase):
    """
    Трехмерный параллелепипед x_min..x_max × y_min..y_max × z_min..z_max
    """

    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        super().__init__(dim=3)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    def sample_interior(self, n_points, method="random", rng=None):
        """Генерирует точки внутри параллелепипеда"""
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

        raise ValueError(f"Unknown method: {method}")

    def sample_boundary(self, n_points, method="random", rng=None, return_faces=False):
        """Генерирует точки на границе параллелепипеда и нормали."""
        if n_points < 6:
            raise ValueError("n_points должен быть >= 6")

        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method != "random":
            raise ValueError(f"Неизвестный метод: {method}")

        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        dz = self.z_max - self.z_min

        # Площади граней: bottom, top, front, back, left, right
        areas = jnp.array([dx*dy, dx*dy, dx*dz, dx*dz, dy*dz, dy*dz])
        
        # Каждая грань получает минимум 1 точку, остаток пропорционально площадям
        n_face = jnp.round((areas / jnp.sum(areas)) * (n_points - 6)).astype(int) + 1
        diff = n_points - jnp.sum(n_face)
        if diff != 0:
            idx = jnp.argmax(areas)
            n_face = n_face.at[idx].add(diff)

        keys = jax.random.split(rng, 6)
        face_names = ['bottom', 'top', 'front', 'back', 'left', 'right']
        all_points = []
        all_normals = []
        faces_data = {name: ([], []) for name in face_names}

        # bottom (z = z_min)
        n = int(n_face[0])
        x = jax.random.uniform(keys[0], (n, 1), minval=self.x_min, maxval=self.x_max)
        y = jax.random.uniform(keys[1], (n, 1), minval=self.y_min, maxval=self.y_max)
        z = jnp.full((n, 1), self.z_min)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['bottom'][0].append(pts)
        faces_data['bottom'][1].append(norms)

        # top (z = z_max)
        n = int(n_face[1])
        x = jax.random.uniform(keys[2], (n, 1), minval=self.x_min, maxval=self.x_max)
        y = jax.random.uniform(keys[3], (n, 1), minval=self.y_min, maxval=self.y_max)
        z = jnp.full((n, 1), self.z_max)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['top'][0].append(pts)
        faces_data['top'][1].append(norms)

        # front (y = y_min)
        n = int(n_face[2])
        x = jax.random.uniform(keys[4], (n, 1), minval=self.x_min, maxval=self.x_max)
        z = jax.random.uniform(keys[5], (n, 1), minval=self.z_min, maxval=self.z_max)
        y = jnp.full((n, 1), self.y_min)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([0.0, -1.0, 0.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['front'][0].append(pts)
        faces_data['front'][1].append(norms)

        # back (y = y_max)
        n = int(n_face[3])
        x = jax.random.uniform(keys[0], (n, 1), minval=self.x_min, maxval=self.x_max)
        z = jax.random.uniform(keys[1], (n, 1), minval=self.z_min, maxval=self.z_max)
        y = jnp.full((n, 1), self.y_max)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([0.0, 1.0, 0.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['back'][0].append(pts)
        faces_data['back'][1].append(norms)

        # left (x = x_min)
        n = int(n_face[4])
        y = jax.random.uniform(keys[2], (n, 1), minval=self.y_min, maxval=self.y_max)
        z = jax.random.uniform(keys[3], (n, 1), minval=self.z_min, maxval=self.z_max)
        x = jnp.full((n, 1), self.x_min)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([-1.0, 0.0, 0.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['left'][0].append(pts)
        faces_data['left'][1].append(norms)

        # right (x = x_max)
        n = int(n_face[5])
        y = jax.random.uniform(keys[4], (n, 1), minval=self.y_min, maxval=self.y_max)
        z = jax.random.uniform(keys[5], (n, 1), minval=self.z_min, maxval=self.z_max)
        x = jnp.full((n, 1), self.x_max)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['right'][0].append(pts)
        faces_data['right'][1].append(norms)

        if return_faces:
            return {name: (jnp.vstack(faces_data[name][0]), jnp.vstack(faces_data[name][1])) 
                    for name in face_names}
        else:
            return jnp.vstack(all_points), jnp.vstack(all_normals)

    def generate_collocation(
        self,
        n_interior,
        n_boundary,
        method_interior="random",
        method_boundary="random",
        rng=None,
    ):
        """Генерирует внутренние и граничные точки"""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary, _ = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])
        
        # ===== НОВЫЕ МЕТОДЫ =====
    def get_boundary_names(self) -> List[str]:
        return ['left', 'right', 'front', 'back', 'bottom', 'top']

    def _generate_boundary_points(self, name: str, n: int, method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if method == "random":
            if name == 'left':
                fixed = self.x_min
                y_vals = jax.random.uniform(rng, (n, 1), minval=self.y_min, maxval=self.y_max)
                z_vals = jax.random.uniform(rng, (n, 1), minval=self.z_min, maxval=self.z_max)
                pts = jnp.hstack([jnp.full((n, 1), fixed), y_vals, z_vals])
                normal = jnp.array([-1.0, 0.0, 0.0])
            elif name == 'right':
                fixed = self.x_max
                y_vals = jax.random.uniform(rng, (n, 1), minval=self.y_min, maxval=self.y_max)
                z_vals = jax.random.uniform(rng, (n, 1), minval=self.z_min, maxval=self.z_max)
                pts = jnp.hstack([jnp.full((n, 1), fixed), y_vals, z_vals])
                normal = jnp.array([1.0, 0.0, 0.0])
            elif name == 'front':
                fixed = self.y_min
                x_vals = jax.random.uniform(rng, (n, 1), minval=self.x_min, maxval=self.x_max)
                z_vals = jax.random.uniform(rng, (n, 1), minval=self.z_min, maxval=self.z_max)
                pts = jnp.hstack([x_vals, jnp.full((n, 1), fixed), z_vals])
                normal = jnp.array([0.0, -1.0, 0.0])
            elif name == 'back':
                fixed = self.y_max
                x_vals = jax.random.uniform(rng, (n, 1), minval=self.x_min, maxval=self.x_max)
                z_vals = jax.random.uniform(rng, (n, 1), minval=self.z_min, maxval=self.z_max)
                pts = jnp.hstack([x_vals, jnp.full((n, 1), fixed), z_vals])
                normal = jnp.array([0.0, 1.0, 0.0])
            elif name == 'bottom':
                fixed = self.z_min
                x_vals = jax.random.uniform(rng, (n, 1), minval=self.x_min, maxval=self.x_max)
                y_vals = jax.random.uniform(rng, (n, 1), minval=self.y_min, maxval=self.y_max)
                pts = jnp.hstack([x_vals, y_vals, jnp.full((n, 1), fixed)])
                normal = jnp.array([0.0, 0.0, -1.0])
            elif name == 'top':
                fixed = self.z_max
                x_vals = jax.random.uniform(rng, (n, 1), minval=self.x_min, maxval=self.x_max)
                y_vals = jax.random.uniform(rng, (n, 1), minval=self.y_min, maxval=self.y_max)
                pts = jnp.hstack([x_vals, y_vals, jnp.full((n, 1), fixed)])
                normal = jnp.array([0.0, 0.0, 1.0])
            else:
                raise ValueError(f"Unknown boundary name '{name}' for Box3D")
            norms = jnp.tile(normal.reshape(1, 3), (n, 1))
            return pts, norms
        else:
            raise NotImplementedError("Uniform method not implemented for Box3D")

    def sample_boundary_by_name(self, n_points_dict: Dict[str, int], method="random", rng=None) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        result = {}
        keys = jax.random.split(rng, 6) if method == "random" and rng is not None else [None] * 6
        for i, name in enumerate(self.get_boundary_names()):
            n = n_points_dict.get(name, 0)
            if n > 0:
                pts, norms = self._generate_boundary_points(name, n, method, keys[i])
                result[name] = (pts, norms)
        return result

    def sample_interface(self, other_geom: 'GeometryBase', n_points: int,
                         self_interface_name: str, other_interface_name: str,
                         method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not isinstance(other_geom, Box3D):
            raise TypeError("other_geom must be Box3D")
        pts, normals_self = self._generate_boundary_points(self_interface_name, n_points, method, rng)
        return pts, normals_self

    @property
    def bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.array([self.x_min, self.y_min, self.z_min]), jnp.array([self.x_max, self.y_max, self.z_max])
        

class Sphere(GeometryBase):
    """
    Трехмерная сфера с центром в (cx, cy, cz) и радиусом R
    """

    def __init__(self, cx=0.0, cy=0.0, cz=0.0, radius=1.0):
        super().__init__(dim=3)
        self.cx = float(cx)
        self.cy = float(cy)
        self.cz = float(cz)
        self.radius = float(radius)

    def sample_interior(self, n_points, method="random", rng=None):
        """Генерирует точки внутри сферы"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == "random":
            keys = jax.random.split(rng, 3)
            # Равномерное распределение в сфере через сферические координаты
            r = self.radius * jnp.cbrt(jax.random.uniform(keys[0], (n_points, 1)))
            theta = 2 * jnp.pi * jax.random.uniform(keys[1], (n_points, 1))
            phi = jnp.arccos(2 * jax.random.uniform(keys[2], (n_points, 1)) - 1)

            x = self.cx + r * jnp.sin(phi) * jnp.cos(theta)
            y = self.cy + r * jnp.sin(phi) * jnp.sin(theta)
            z = self.cz + r * jnp.cos(phi)
            return jnp.hstack([x, y, z])

        raise ValueError(f"Unknown method: {method}")

    def sample_boundary(self, n_points=100, method="random", rng=None):
        """
        Генерирует точки на поверхности сферы и нормали.

        Returns:
            tuple[jax.Array, jax.Array]:
                - points: массив формы (n_points, 3)
                - normals: массив единичных нормалей формы (n_points, 3)
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == "random":
            keys = jax.random.split(rng, 2)
            theta = 2 * jnp.pi * jax.random.uniform(keys[0], (n_points,))
            phi = jnp.arccos(2 * jax.random.uniform(keys[1], (n_points,)) - 1)

            sin_phi = jnp.sin(phi)
            cos_phi = jnp.cos(phi)
            sin_theta = jnp.sin(theta)
            cos_theta = jnp.cos(theta)

            x = self.cx + self.radius * sin_phi[:, None] * cos_theta[:, None]
            y = self.cy + self.radius * sin_phi[:, None] * sin_theta[:, None]
            z = self.cz + self.radius * cos_phi[:, None]
            
            points = jnp.hstack([x, y, z])
            
            # Нормали направлены радиально наружу
            normals = jnp.hstack([
                sin_phi[:, None] * cos_theta[:, None],
                sin_phi[:, None] * sin_theta[:, None],
                cos_phi[:, None]
            ])
            
            return points, normals

        raise ValueError(f"Unknown method: {method}")

    def generate_collocation(
        self,
        n_interior,
        n_boundary,
        method_interior="random",
        method_boundary="random",
        rng=None,
    ):
        """Генерирует внутренние и граничные точки"""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary, _ = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])
        
    # ===== НОВЫЕ МЕТОДЫ =====
    def get_boundary_names(self) -> List[str]:
        return ['outer']

    def _generate_boundary_points(self, name: str, n: int, method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if name != 'outer':
            raise ValueError(f"Sphere only has boundary 'outer', got '{name}'")
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        theta = 2 * jnp.pi * jax.random.uniform(keys[0], (n, 1))
        phi = jnp.arccos(2 * jax.random.uniform(keys[1], (n, 1)) - 1)
        x = self.cx + self.radius * jnp.sin(phi) * jnp.cos(theta)
        y = self.cy + self.radius * jnp.sin(phi) * jnp.sin(theta)
        z = self.cz + self.radius * jnp.cos(phi)
        pts = jnp.hstack([x, y, z])
        normals = jnp.hstack([
            jnp.sin(phi) * jnp.cos(theta),
            jnp.sin(phi) * jnp.sin(theta),
            jnp.cos(phi)
        ])
        return pts, normals

    def sample_boundary_by_name(self, n_points_dict: Dict[str, int], method="random", rng=None) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        result = {}
        for name, n in n_points_dict.items():
            if n > 0:
                pts, norms = self._generate_boundary_points(name, n, method, rng)
                result[name] = (pts, norms)
        return result

    def sample_interface(self, other_geom: 'GeometryBase', n_points: int,
                         self_interface_name: str, other_interface_name: str,
                         method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not isinstance(other_geom, Sphere):
            raise TypeError("other_geom must be Sphere")
        if self_interface_name != 'outer' or other_interface_name != 'outer':
            raise ValueError("Sphere interface only supported for 'outer' boundary")
        pts, normals_self = self._generate_boundary_points('outer', n_points, method, rng)
        return pts, normals_self

    @property
    def bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        R = self.radius
        return jnp.array([self.cx - R, self.cy - R, self.cz - R]), jnp.array([self.cx + R, self.cy + R, self.cz + R])
        

class Annulus2D(GeometryBase):
    """
    2D кольцо (полый круг) с внутренним и внешним радиусами.

    Параметры:
        center: центр (x, y)
        R_outer: внешний радиус (> 0)
        R_inner: внутренний радиус (> 0)
    """

    def __init__(
        self, center: tuple[float, float], R_outer: float, R_inner: float = 0.0
    ):
        if R_outer <= 0:
            raise ValueError("R_outer должен быть > 0")
        if R_inner <= 0:
            raise ValueError("R_inner должен быть > 0")
        if R_inner >= R_outer:
            raise ValueError("R_inner должен быть < R_outer")

        super().__init__(dim=2)
        self.center = jnp.array(center, dtype=jnp.float32)
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)
        self.is_hollow = R_inner > 0

    def sample_interior(
        self, n_points: int, rng: jax.Array | None = None
    ) -> jnp.ndarray:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        r_max = self.R_outer
        r_min = self.R_inner

        # Генерируем u и theta из одного ключа
        key1, key2 = jax.random.split(rng, 2)
        u = jax.random.uniform(key1, (n_points,))
        theta = jax.random.uniform(key2, (n_points,), minval=0, maxval=2 * jnp.pi)

        if self.is_hollow:
            r = jnp.sqrt(r_min**2 + u * (r_max**2 - r_min**2))
        else:
            r = jnp.sqrt(u) * r_max

        x = self.center[0] + r * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(theta)

        points = jnp.column_stack([x, y])

        # Перемешиваем точки
        key3, _ = jax.random.split(rng, 2)
        points = jax.random.permutation(key3, points, axis=0)

        return points

    def sample_boundary(
        self, n_points: int, rng: jax.Array | None = None, return_edges: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray] | dict:
        """
        Генерирует точки на границе кольца и нормали.

        Args:
            n_points: общее количество точек (распределяются поровну между границами)
            rng: ключ PRNG
            return_edges: если True, возвращает словарь с ключами 'outer' и 'inner'

        Returns:
            Если return_edges=False: (points, normals) – объединённые массивы
            Если return_edges=True: словарь с ключами 'outer' и 'inner'
        """
        if n_points <= 0:
            raise ValueError("Количество точек должно быть положительным")

        if rng is None:
            rng = jax.random.PRNGKey(1)

        keys = jax.random.split(rng, 2)

        # Если кольцо не полое - только внешняя граница
        if not self.is_hollow:
            theta = jax.random.uniform(
                keys[0], (n_points,), minval=0, maxval=2 * jnp.pi
            )
            x = self.center[0] + self.R_outer * jnp.cos(theta)
            y = self.center[1] + self.R_outer * jnp.sin(theta)
            points = jnp.column_stack([x, y])
            normals = jnp.column_stack([jnp.cos(theta), jnp.sin(theta)])
            
            if return_edges:
                return {'outer': (points, normals)}
            else:
                return points, normals

        # Полое кольцо: распределяем точки поровну
        n_outer = n_points // 2
        n_inner = n_points - n_outer  # остаток отдаём внутренней границе

        # Внешняя граница
        theta_out = jax.random.uniform(
            keys[0], (n_outer,), minval=0, maxval=2 * jnp.pi
        )
        x_out = self.center[0] + self.R_outer * jnp.cos(theta_out)
        y_out = self.center[1] + self.R_outer * jnp.sin(theta_out)
        points_outer = jnp.column_stack([x_out, y_out])
        normals_outer = jnp.column_stack([jnp.cos(theta_out), jnp.sin(theta_out)])

        # Внутренняя граница
        theta_in = jax.random.uniform(
            keys[1], (n_inner,), minval=0, maxval=2 * jnp.pi
        )
        x_in = self.center[0] + self.R_inner * jnp.cos(theta_in)
        y_in = self.center[1] + self.R_inner * jnp.sin(theta_in)
        points_inner = jnp.column_stack([x_in, y_in])
        # Нормаль направлена внутрь полости (наружу из материала)
        normals_inner = jnp.column_stack([-jnp.cos(theta_in), -jnp.sin(theta_in)])

        if return_edges:
            return {
                'outer': (points_outer, normals_outer),
                'inner': (points_inner, normals_inner)
            }
        else:
            points = jnp.vstack([points_outer, points_inner])
            normals = jnp.vstack([normals_outer, normals_inner])
            return points, normals

    def is_inside(self, points: jnp.ndarray) -> jnp.ndarray:
        r = jnp.linalg.norm(points - self.center, axis=1)
        return (r <= self.R_outer) & (r >= self.R_inner)
        
        # ===== НОВЫЕ МЕТОДЫ =====
    def get_boundary_names(self) -> List[str]:
        if self.is_hollow:
            return ['outer', 'inner']
        else:
            return ['outer']

    def _generate_boundary_points(self, name: str, n: int, method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if name == 'outer':
            R = self.R_outer
            normal_sign = 1.0
        elif name == 'inner':
            if not self.is_hollow:
                raise ValueError("Inner boundary not available (solid disk)")
            R = self.R_inner
            normal_sign = -1.0
        else:
            raise ValueError(f"Unknown boundary name '{name}' for Annulus2D")
        theta = jax.random.uniform(rng, (n,), minval=0, maxval=2 * jnp.pi)
        x = self.center[0] + R * jnp.cos(theta)
        y = self.center[1] + R * jnp.sin(theta)
        pts = jnp.column_stack([x, y])
        normals = normal_sign * jnp.column_stack([jnp.cos(theta), jnp.sin(theta)])
        return pts, normals

    def sample_boundary_by_name(self, n_points_dict: Dict[str, int], method="random", rng=None) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        result = {}
        for name, n in n_points_dict.items():
            if n > 0:
                pts, norms = self._generate_boundary_points(name, n, method, rng)
                result[name] = (pts, norms)
        return result

    def sample_interface(self, other_geom: 'GeometryBase', n_points: int,
                         self_interface_name: str, other_interface_name: str,
                         method="random", rng=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not isinstance(other_geom, Annulus2D):
            raise TypeError("other_geom must be Annulus2D")
        pts, normals_self = self._generate_boundary_points(self_interface_name, n_points, method, rng)
        return pts, normals_self

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        min_corner = self.center - self.R_outer
        max_corner = self.center + self.R_outer
        return min_corner, max_corner
    
class HollowCylinder3D(GeometryBase):
    """
    3D полый цилиндр (труба) с внутренним и внешним радиусами.

    Параметры:
        center_base: центр основания (x, y, z)
        radius_outer: внешний радиус (> 0)
        radius_inner: внутренний радиус (> 0, строго полый)
        height: высота цилиндра
        axis: ось направления (по умолчанию Z)
    """

    def __init__(
        self,
        center_base: tuple[float, float, float],
        radius_outer: float,
        radius_inner: float,
        height: float,
        axis: tuple[float, float, float] = (0, 0, 1),
    ):
        if radius_outer <= 0:
            raise ValueError("radius_outer должен быть > 0")
        if radius_inner <= 0:
            raise ValueError("radius_inner должен быть > 0 (строго полый)")
        if radius_inner >= radius_outer:
            raise ValueError("radius_inner должен быть < radius_outer")
        if height <= 0:
            raise ValueError("height должен быть > 0")

        super().__init__(dim=3)
        self.center_base = jnp.array(center_base, dtype=jnp.float64)
        self.radius_outer = float(radius_outer)
        self.radius_inner = float(radius_inner)
        self.height = float(height)
        self.axis = jnp.array(axis, dtype=jnp.float64)
        self.axis = self.axis / jnp.linalg.norm(self.axis)

    # ------------------------------------------------------------------ tags

    def get_boundary_tags(self) -> dict[str, str]:
        return {
            "outer": "exterior_boundary",
            "inner": "interface",
            "bottom": "exterior_boundary",
            "top": "exterior_boundary",
        }

    # --------------------------------------------------------------- helpers

    def _cylinder_basis(self):
        """Возвращает (ez, er_ref, etheta_ref) — ортонормированный базис цилиндра."""
        ez = self.axis
        if jnp.abs(ez[2]) < 0.9:
            er_ref = jnp.cross(ez, jnp.array([0.0, 0.0, 1.0]))
        else:
            er_ref = jnp.cross(ez, jnp.array([1.0, 0.0, 0.0]))
        er_ref = er_ref / jnp.linalg.norm(er_ref)
        etheta_ref = jnp.cross(ez, er_ref)
        return ez, er_ref, etheta_ref

    def _cylindrical_to_cartesian(
        self, r: jnp.ndarray, theta: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Преобразование цилиндрических координат в декартовы."""
        ez, er_ref, etheta_ref = self._cylinder_basis()

        x_local = (
            (r * jnp.cos(theta))[:, None] * er_ref
            + (r * jnp.sin(theta))[:, None] * etheta_ref
            + z[:, None] * ez
        )
        return self.center_base + x_local

    # ------------------------------------------------------------ sample points

    def sample_interior(
        self, n_points: int, rng: jax.Array | None = None
    ) -> jnp.ndarray:
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

    # ---------------------------------------------------- boundary by name

    def get_boundary_names(self) -> List[str]:
        return ['outer', 'inner', 'bottom', 'top']

    def _generate_boundary_points(self, name: str, n: int,
                                  method="random", rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)

        ez, er_ref, etheta_ref = self._cylinder_basis()

        if name == 'outer':
            theta = jax.random.uniform(rng, (n,), minval=0.0, maxval=2 * jnp.pi)
            z = jax.random.uniform(rng, (n,), minval=0.0, maxval=self.height)
            r = jnp.full(n, self.radius_outer)

            pts = self._cylindrical_to_cartesian(r, theta, z)
            norms = (jnp.cos(theta)[:, None] * er_ref
                     + jnp.sin(theta)[:, None] * etheta_ref)
            return pts, norms

        elif name == 'inner':
            theta = jax.random.uniform(rng, (n,), minval=0.0, maxval=2 * jnp.pi)
            z = jax.random.uniform(rng, (n,), minval=0.0, maxval=self.height)
            r = jnp.full(n, self.radius_inner)

            pts = self._cylindrical_to_cartesian(r, theta, z)
            norms = -(jnp.cos(theta)[:, None] * er_ref
                      + jnp.sin(theta)[:, None] * etheta_ref)
            return pts, norms

        elif name == 'bottom':
            u = jax.random.uniform(rng, (n,))
            theta = jax.random.uniform(rng, (n,), minval=0.0, maxval=2 * jnp.pi)
            r = jnp.sqrt(self.radius_inner**2
                         + u * (self.radius_outer**2 - self.radius_inner**2))
            z = jnp.zeros(n)

            pts = self._cylindrical_to_cartesian(r, theta, z)
            norms = jnp.tile(-self.axis, (n, 1))
            return pts, norms

        elif name == 'top':
            u = jax.random.uniform(rng, (n,))
            theta = jax.random.uniform(rng, (n,), minval=0.0, maxval=2 * jnp.pi)
            r = jnp.sqrt(self.radius_inner**2
                         + u * (self.radius_outer**2 - self.radius_inner**2))
            z = jnp.full(n, self.height)

            pts = self._cylindrical_to_cartesian(r, theta, z)
            norms = jnp.tile(self.axis, (n, 1))
            return pts, norms

        else:
            raise ValueError(f"Unknown boundary name '{name}' for HollowCylinder3D")

    def sample_boundary_by_name(self, n_points_dict: Dict[str, int],
                                method="random", rng=None):
        result = {}
        keys = jax.random.split(rng, 4) if method == "random" and rng is not None else [None] * 4
        for i, name in enumerate(self.get_boundary_names()):
            n = n_points_dict.get(name, 0)
            if n > 0:
                pts, norms = self._generate_boundary_points(name, n, method, keys[i])
                result[name] = (pts, norms)
        return result

    # ------------------------------------------------------------ sample_boundary

    def sample_boundary(
        self, n_points: int, rng: jax.Array | None = None, tags: list | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(1)

        keys = jax.random.split(rng, 8)
        points_list = []
        normals_list = []

        n_surfaces = 4
        n_per_surface = n_points // n_surfaces
        remainder = n_points - n_per_surface * n_surfaces

        # 1. outer
        if tags is None or "outer" in tags:
            n_outer = n_per_surface + (remainder if remainder > 0 else 0)
            pts, norms = self._generate_boundary_points('outer', n_outer, rng=keys[0])
            points_list.append(pts)
            normals_list.append(norms)

        # 2. inner
        if tags is None or "inner" in tags:
            pts, norms = self._generate_boundary_points('inner', n_per_surface, rng=keys[2])
            points_list.append(pts)
            normals_list.append(norms)

        # 3. bottom
        if tags is None or "bottom" in tags:
            pts, norms = self._generate_boundary_points('bottom', n_per_surface, rng=keys[4])
            points_list.append(pts)
            normals_list.append(norms)

        # 4. top
        if tags is None or "top" in tags:
            pts, norms = self._generate_boundary_points('top', n_per_surface, rng=keys[6])
            points_list.append(pts)
            normals_list.append(norms)

        return jnp.vstack(points_list), jnp.vstack(normals_list)

    # ------------------------------------------------------------ is_inside

    def is_inside(self, points: jnp.ndarray) -> jnp.ndarray:
        vec = points - self.center_base
        z_coord = jnp.dot(vec, self.axis)
        r_vec = vec - z_coord[:, None] * self.axis
        r = jnp.linalg.norm(r_vec, axis=1)

        return (
            (r <= self.radius_outer)
            & (r >= self.radius_inner)
            & (z_coord >= 0)
            & (z_coord <= self.height)
        )

    # ------------------------------------------------------------ interface

    def sample_interface(self, other_geom: 'GeometryBase', n_points: int,
                         self_interface_name: str, other_interface_name: str,
                         method="random", rng=None):
        if not isinstance(other_geom, HollowCylinder3D):
            raise TypeError("other_geom must be HollowCylinder3D")
        pts, normals_self = self._generate_boundary_points(
            self_interface_name, n_points, method, rng)
        return pts, normals_self

    # ------------------------------------------------------------ bounds

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        R = self.radius_outer
        H = self.height
        min_corner = self.center_base - jnp.array([R, R, 0.0])
        max_corner = self.center_base + jnp.array([R, R, H])
        return min_corner, max_corner

class HollowSphere3D(GeometryBase):
    """
    3D полая сфера с внутренним и внешним радиусами.

    Параметры:
        center: центр сферы (x, y, z)
        R_outer: внешний радиус (> 0)
        R_inner: внутренний радиус (> 0, строго полый)
    """

    def __init__(
        self, center: tuple[float, float, float], R_outer: float, R_inner: float
    ):
        if R_outer <= 0:
            raise ValueError("R_outer должен быть > 0")
        if R_inner <= 0:
            raise ValueError("R_inner должен быть > 0 (строго полый)")
        if R_inner >= R_outer:
            raise ValueError("R_inner должен быть < R_outer")

        super().__init__(dim=3)
        self.center = jnp.array(center, dtype=jnp.float64)
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)

    # ------------------------------------------------------------------ tags

    def get_boundary_tags(self) -> dict[str, str]:
        return {"outer": "exterior_boundary", "inner": "interface"}

    # ------------------------------------------------------------ sample points

    def sample_interior(
        self, n_points: int, rng: jax.Array | None = None
    ) -> jnp.ndarray:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        keys = jax.random.split(rng, 3)
        u = jax.random.uniform(keys[0], (n_points,))
        v = jax.random.uniform(keys[1], (n_points,))
        w = jax.random.uniform(keys[2], (n_points,))

        theta = 2 * jnp.pi * u
        phi = jnp.arccos(2 * v - 1)
        r = (self.R_inner**3 + w * (self.R_outer**3 - self.R_inner**3)) ** (1 / 3)

        x = self.center[0] + r * jnp.sin(phi) * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(phi) * jnp.sin(theta)
        z = self.center[2] + r * jnp.cos(phi)

        return jnp.column_stack([x, y, z])

    # ---------------------------------------------------- boundary by name

    def get_boundary_names(self) -> List[str]:
        return ['outer', 'inner']

    def _generate_boundary_points(self, name: str, n: int,
                                  method="random", rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if name == 'outer':
            R = self.R_outer
            sign = 1.0
        elif name == 'inner':
            R = self.R_inner
            sign = -1.0
        else:
            raise ValueError(f"Unknown boundary name '{name}' for HollowSphere3D")

        keys = jax.random.split(rng, 2)
        theta = 2 * jnp.pi * jax.random.uniform(keys[0], (n,))
        phi = jnp.arccos(2 * jax.random.uniform(keys[1], (n,)) - 1)

        sin_phi = jnp.sin(phi)
        cos_phi = jnp.cos(phi)
        sin_t = jnp.sin(theta)
        cos_t = jnp.cos(theta)

        # Точки
        x = self.center[0] + R * sin_phi * cos_t
        y = self.center[1] + R * sin_phi * sin_t
        z = self.center[2] + R * cos_phi
        pts = jnp.column_stack([x, y, z])

        # Единичная радиальная нормаль (наружу для 'outer', внутрь для 'inner')
        nx = sin_phi * cos_t
        ny = sin_phi * sin_t
        nz = cos_phi
        norms = sign * jnp.column_stack([nx, ny, nz])

        return pts, norms

    def sample_boundary_by_name(self, n_points_dict: Dict[str, int],
                                method="random", rng=None):
        result = {}
        keys = jax.random.split(rng, 2) if method == "random" and rng is not None else [None] * 2
        for i, name in enumerate(self.get_boundary_names()):
            n = n_points_dict.get(name, 0)
            if n > 0:
                pts, norms = self._generate_boundary_points(name, n, method, keys[i])
                result[name] = (pts, norms)
        return result

    # ------------------------------------------------------------ sample_boundary

    def sample_boundary(
        self, n_points: int, rng: jax.Array | None = None, tags: list | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(1)

        keys = jax.random.split(rng, 2)
        points_list = []
        normals_list = []

        n_per_surface = n_points // 2

        # Внешняя сфера
        if tags is None or "outer" in tags:
            pts, norms = self._generate_boundary_points('outer', n_per_surface, rng=keys[0])
            points_list.append(pts)
            normals_list.append(norms)

        # Внутренняя сфера
        if tags is None or "inner" in tags:
            pts, norms = self._generate_boundary_points('inner', n_per_surface, rng=keys[1])
            points_list.append(pts)
            normals_list.append(norms)

        return jnp.vstack(points_list), jnp.vstack(normals_list)

    # ------------------------------------------------------------ is_inside

    def is_inside(self, points: jnp.ndarray) -> jnp.ndarray:
        r = jnp.linalg.norm(points - self.center, axis=1)
        return (r <= self.R_outer) & (r >= self.R_inner)

    # ------------------------------------------------------------ interface

    def sample_interface(self, other_geom: 'GeometryBase', n_points: int,
                         self_interface_name: str, other_interface_name: str,
                         method="random", rng=None):
        if not isinstance(other_geom, HollowSphere3D):
            raise TypeError("other_geom must be HollowSphere3D")
        pts, normals_self = self._generate_boundary_points(
            self_interface_name, n_points, method, rng)
        return pts, normals_self
    # ------------------------------------------------------------ bounds

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        R = self.R_outer
        min_corner = self.center - R
        max_corner = self.center + R
        return min_corner, max_corner
    