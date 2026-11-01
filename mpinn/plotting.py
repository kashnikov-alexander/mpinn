"""
Модуль визуализации для PINN.
Поддерживает 1D, 2D и 3D графики, а также историю обучения.
Работает с JAX и NumPy массивами напрямую.
"""

import os
from typing import Optional, List, Dict, Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .geom import GeometryBase

# ===================== 1D графики (обратно совместимые) =====================
def show_plot(
    x_test: Union[np.ndarray, "jnp.ndarray"],
    T_pred: Union[np.ndarray, "jnp.ndarray"],
    T_exact: Union[np.ndarray, "jnp.ndarray"],
    title: str = "Сравнение PINN и точного решения"
) -> None:
    """Отображает 1D график сравнения."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_test, T_exact, "b-", label="Аналитическое решение", linewidth=2)
    plt.plot(x_test, T_pred, "r:", label="ФИНС", linewidth=6)
    plt.title(title, fontsize=16)
    plt.xlabel("x, м", fontsize=14)
    plt.ylabel("T, К", fontsize=14)
    plt.legend(loc="best", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(x_test.min(), x_test.max())
    plt.tight_layout()
    plt.show()

def save_plot(
    x_test: Union[np.ndarray, "jnp.ndarray"],
    T_pred: Union[np.ndarray, "jnp.ndarray"],
    T_exact: Union[np.ndarray, "jnp.ndarray"],
    save_path: str = "",
    title: str = "Сравнение PINN и точного решения"
) -> None:
    """Сохраняет 1D график сравнения."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_test, T_exact, "b-", label="Аналитическое решение", linewidth=2)
    plt.plot(x_test, T_pred, "r:", label="ФИНС", linewidth=6)
    plt.title(title, fontsize=16)
    plt.xlabel("x, м", fontsize=14)
    plt.ylabel("T, К", fontsize=14)
    plt.legend(loc="best", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(x_test.min(), x_test.max())
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=72, bbox_inches="tight")
    plt.close()

def show_history(
    history,
    labels=None,
    logy=True,
    figsize=(12, 7),
    title="История обучения",
    xlabel="Эпохи",
    ylabel="Потери",
    save_path=None,
):
    """Универсальная визуализация истории обучения.

    history: dict[str, list[float] | np.ndarray]
        Должен содержать 'steps' (x-ось). Остальные ключи — что рисовать.
    labels: list[str] | None
        None → все ключи, кроме 'steps'.
        list → только эти ключи, в этом порядке.
    logy: bool
        Логарифмическая шкала по y. Отключайте, если среди значений есть <= 0.
    save_path: str | None
        Если задан — сохранить в файл.

    Returns
    -------
    (fig, ax)
    """
    if "steps" not in history:
        raise KeyError(
            "history должен содержать 'steps'. "
            "fit() всегда его пишет; проверьте, что вы передаёте history, "
            "а не отдельный список."
        )

    steps = np.asarray(history["steps"])
    n = len(steps)

    if labels is None:
        labels = [k for k in history if k != "steps"]

    if not labels:
        raise ValueError("Нет ключей для отображения.")

    fig, ax = plt.subplots(figsize=figsize)

    for label in labels:
        if label not in history:
            raise KeyError(
                f"history не содержит ключ {label!r}. "
                f"Доступные: {sorted(k for k in history if k != 'steps')}"
            )
        values = np.asarray(history[label])
        if len(values) != n:
            raise ValueError(
                f"Длина history[{label!r}] = {len(values)} "
                f"не совпадает с длиной 'steps' = {n}."
            )
        ax.plot(steps, values, label=label, linewidth=2)

    if logy:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=72, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    plt.show()
    return fig, ax


def show_plot_mpinn(
    domain_data: list,
    exact_fn=None,
    interfaces=None,
    title: str = "MPINN: сравнение с аналитическим решением",
    save_path: str | None = None,
    dpi: int = 100,
    figsize: tuple = (10, 5),
    show: bool = True,
):
    """1D-профиль MPINN по доменам.

    domain_data: list[dict]
        каждый dict: {'x': array (N,), 'T_pred': array (N,),
                      'color': str (опц.), 'label': str (опц.),
                      'T_exact': array (N,) (опц.)}
    exact_fn: callable | None
        Если задан — рисуется сплошной линией на объединённом диапазоне x.
        Вызывается как exact_fn(x.reshape(-1, 1)) -> (N, 1).
    interfaces: list[float] | None
        x-координаты интерфейсов (вертикальные линии).
    save_path: str | None
        Если задан — сохранить. dirname может быть пустым.
    dpi: int
    figsize: tuple
    show: bool
        False — не вызывать plt.show() (для batch-режима).

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # --- Аналитическое решение на объединённом диапазоне ---
    if exact_fn is not None:
        x_min = min(float(np.asarray(d["x"]).min()) for d in domain_data)
        x_max = max(float(np.asarray(d["x"]).max()) for d in domain_data)
        x_full = np.linspace(x_min, x_max, 1000)
        T_full = np.asarray(exact_fn(x_full.reshape(-1, 1))).ravel()
        ax.plot(x_full, T_full, "b-",
                label="Аналитическое решение", linewidth=2, zorder=1)

    # --- Домены ---
    for i, d in enumerate(domain_data):
        color = d.get("color", plt.cm.tab10(i))
        label = d.get("label", f"Домен {i}")
        x = np.asarray(d["x"])
        ax.plot(x, np.asarray(d["T_pred"]), ":", color=color,
                label=label, linewidth=5, zorder=2)

        # T_exact по домену — только если exact_fn НЕ задан
        # (иначе дублирование с глобальной линией)
        if exact_fn is None and "T_exact" in d:
            ax.plot(x, np.asarray(d["T_exact"]), "--", color=color,
                    linewidth=1, alpha=0.6, zorder=0)

    # --- Интерфейсы ---
    if interfaces:
        for k, x_int in enumerate(interfaces):
            ax.axvline(x_int, color="gray", linestyle="--",
                       linewidth=1.5, alpha=0.6, zorder=0,
                       label="Интерфейс" if k == 0 else None)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x, м", fontsize=13)
    ax.set_ylabel("T, К", fontsize=13)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig, ax

# ===================== 2D/3D графики =====================
def plot_2d_contour(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    levels: int = 50,
    title: str = "Температурное поле",
    xlabel: str = "x, м",
    ylabel: str = "y, м",
    cmap: str = "jet",
    save_path: Optional[str] = None,
) -> None:
    """Строит контурный график по готовым X, Y, T."""
    plt.figure(figsize=(7, 6))
    contour = plt.contourf(X, Y, T, levels=levels, cmap=cmap)
    plt.colorbar(contour, label="T, К")
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=72, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_2d(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    title: str = "Температурное поле",
    xlabel: str = "x, м",
    ylabel: str = "y, м",
    cmap: str = "jet",
    save_path: Optional[str] = None,
) -> None:
    """Строит пиксельный график по готовым X, Y, T."""
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())

    plt.figure(figsize=(6, 6))
    plt.imshow(
        T.T,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(label="T, К")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=72, bbox_inches="tight")
    plt.show()
    plt.close()

# ===================== 3D графики (с маскированием) =====================

def plot_3d_slices(
    geom: "GeometryBase",
    predict_fn: Callable,
    slice_planes: Optional[List[Dict[str, float]]] = None,
    resolution: int = 60,
    title: str = "Срезы температуры",
    save_path: Optional[str] = None,
) -> None:
    """
    Срезы 3D-геометрии по плоскостям с маскированием вне домена
    (через geom.is_inside, если есть).
    """
    if geom.dim != 3:
        raise ValueError("plot_3d_slices поддерживает только 3D геометрию.")

    min_corner, max_corner = geom.bounds
    x_min, y_min, z_min = [float(v) for v in min_corner]
    x_max, y_max, z_max = [float(v) for v in max_corner]

    has_mask = hasattr(geom, 'is_inside')

    if slice_planes is None:
        slice_planes = [
            {"axis": "x", "value": (x_min + x_max) / 2},
            {"axis": "y", "value": (y_min + y_max) / 2},
            {"axis": "z", "value": (z_min + z_max) / 2},
        ]

    n_planes = len(slice_planes)
    fig, axes = plt.subplots(1, n_planes, figsize=(5 * n_planes, 4))
    if n_planes == 1:
        axes = [axes]

    for ax, plane in zip(axes, slice_planes):
        axis = plane["axis"]
        value = plane["value"]

        if axis == "x":
            a = np.linspace(y_min, y_max, resolution)
            b = np.linspace(z_min, z_max, resolution)
            A, B = np.meshgrid(a, b, indexing="ij")
            pts = np.column_stack([np.full(resolution**2, value),
                                   A.ravel(), B.ravel()])
            xlabel, ylabel = "y, м", "z, м"
        elif axis == "y":
            a = np.linspace(x_min, x_max, resolution)
            b = np.linspace(z_min, z_max, resolution)
            A, B = np.meshgrid(a, b, indexing="ij")
            pts = np.column_stack([A.ravel(),
                                   np.full(resolution**2, value),
                                   B.ravel()])
            xlabel, ylabel = "x, м", "z, м"
        elif axis == "z":
            a = np.linspace(x_min, x_max, resolution)
            b = np.linspace(y_min, y_max, resolution)
            A, B = np.meshgrid(a, b, indexing="ij")
            pts = np.column_stack([A.ravel(), B.ravel(),
                                   np.full(resolution**2, value)])
            xlabel, ylabel = "x, м", "y, м"
        else:
            raise ValueError(f"Неизвестная ось: {axis}")

        T = np.asarray(predict_fn(pts)).ravel()

        if has_mask:
            mask = np.asarray(geom.is_inside(jnp.asarray(pts)))
            T = np.where(mask, T, np.nan)

        T = T.reshape(resolution, resolution)

        contour = ax.contourf(A, B, T, levels=50, cmap="jet")
        plt.colorbar(contour, ax=ax, label="T, К")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.set_title(f"{axis} = {value:.3f}")

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_3d_isosurface(
    geom: "GeometryBase",
    predict_fn: Callable,
    resolution: int = 40,
    n_surfaces: int = 10,
    title: str = "Изоповерхности T",
    save_path: Optional[str] = None,
) -> None:
    """3D изоповерхности T через Plotly. Маска через geom.is_inside."""
    if not HAS_PLOTLY:
        raise ImportError("pip install plotly")
    if geom.dim != 3:
        raise ValueError("plot_3d_isosurface для 3D")

    min_c, max_c = geom.bounds
    x = np.linspace(float(min_c[0]), float(max_c[0]), resolution)
    y = np.linspace(float(min_c[1]), float(max_c[1]), resolution)
    z = np.linspace(float(min_c[2]), float(max_c[2]), resolution)
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")

    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    T = np.asarray(predict_fn(pts)).ravel()

    if hasattr(geom, "is_inside"):
        mask = np.asarray(geom.is_inside(jnp.asarray(pts)))
        T = np.where(mask, T, np.nan)

    T_3d = T.reshape(XX.shape)

    fig = go.Figure(data=go.Isosurface(
        x=XX.ravel(), y=YY.ravel(), z=ZZ.ravel(),
        value=T_3d.ravel(),
        isomin=float(np.nanmin(T_3d)),
        isomax=float(np.nanmax(T_3d)),
        surface_count=n_surfaces,
        colorscale="Jet",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x, м",
                   yaxis_title="y, м",
                   zaxis_title="z, м",
                   aspectmode="data"),
        width=900, height=700,
    )
    fig.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
        print(f"Сохранено: {save_path}")



def plot_3d_surface_plotly(
    X: Union[np.ndarray, "jnp.ndarray"],
    Y: Union[np.ndarray, "jnp.ndarray"],
    Z: Union[np.ndarray, "jnp.ndarray"],
    title: str = "3D Температурное поле",
    labels: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Создает интерактивный 3D график поверхности с помощью Plotly.
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly не установлен. Установите его: pip install plotly"
        )

    # Приводим к NumPy для надёжности (plotly лучше работает с numpy)
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Температура'}

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='jet')])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=labels.get('x', 'X'),
            yaxis_title=labels.get('y', 'Y'),
            zaxis_title=labels.get('z', 'Z'),
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
        print(f"График сохранён: {save_path}")

