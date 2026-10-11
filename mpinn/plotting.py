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


def show_history(history, log_interval=1, save_path=None):
    """
    Универсальная визуализация истории обучения.
    Работает и для PINN (pde, bc_total, bc_i), и для MPINN (pde_total, bc_total, itf_total).
    """
    # --- длина истории ---
    sample_key = next((k for k, v in history.items() if len(v) > 0), None)
    if sample_key is None:
        print("Нет данных для отображения истории.")
        return

    n = len(history[sample_key])
    epochs = [i * log_interval for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 7))

    # --- PDE ---
    if "pde_total" in history:
        ax.semilogy(epochs, history["pde_total"], "b",
                    label="PDE", linewidth=2)
    elif "pde" in history:
        ax.semilogy(epochs, history["pde"], "b",
                    label="PDE", linewidth=2)

    # --- ГУ (суммарный BC) ---
    if "bc_total" in history:
        ax.semilogy(epochs, history["bc_total"], "r--",
                    label="ГУ", linewidth=2)
    else:
        # fallback: сумма bc_i, если bc_total нет
        bc_keys = sorted(k for k in history if k.startswith("bc_"))
        if bc_keys:
            bc_sum = [sum(vals) for vals in zip(*[history[k] for k in bc_keys])]
            ax.semilogy(epochs, bc_sum, "r--",
                        label="ГУ", linewidth=2)

    # --- Интерфейс (только MPINN) ---
    if "itf_total" in history:
        ax.semilogy(epochs, history["itf_total"], "m--",
                    label="Интерфейс", linewidth=2)
    else:
        itf_keys = sorted(k for k in history if k.startswith("interface_"))
        if itf_keys:
            itf_sum = [sum(vals) for vals in zip(*[history[k] for k in itf_keys])]
            ax.semilogy(epochs, itf_sum, "m--",
                        label="Интерфейс", linewidth=2)

    # --- Общая ---
    if "total_loss" in history:
        ax.semilogy(epochs, history["total_loss"], "g--",
                    label="Общая", linewidth=2, alpha=0.7)

    ax.set_title("История обучения")
    ax.set_xlabel("Эпохи")
    ax.set_ylabel("Потери")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=72, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    plt.show()
    plt.close(fig)

def show_plot_mpinn(
    domain_data: list[dict],
    exact_fn=None,
    interfaces: list[float] | None = None,
    title: str = "MPINN: сравнение с аналитическим решением",
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Аналитическое решение ---
    if exact_fn is not None:
        x_min = min(d['x'].min() for d in domain_data)
        x_max = max(d['x'].max() for d in domain_data)
        x_full = np.linspace(x_min, x_max, 1000)
        T_full = exact_fn(x_full.reshape(-1, 1)).ravel()
        ax.plot(x_full, T_full, 'b-', label='Аналитическое', linewidth=2, zorder=1)

    # --- Домены ---
    default_colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']
    for i, d in enumerate(domain_data):
        color = d.get('color', default_colors[i % len(default_colors)])
        label = d.get('label', f'Домен {i}')
        ax.plot(d['x'], d['T_pred'], ':', color=color,
                label=label, linewidth=5, zorder=2)

        if 'T_exact' in d:
            ax.plot(d['x'], d['T_exact'], '--', color=color,
                    linewidth=1, alpha=0.5, zorder=0)

    # --- Интерфейсы ---
    if interfaces:
        for x_int in interfaces:
            ax.axvline(x_int, color='gray', linestyle='--',
                       linewidth=1.5, alpha=0.6, zorder=0)
        ax.axvline(np.nan, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.6, label='Интерфейс')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x, м", fontsize=13)
    ax.set_ylabel("T, К", fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # --- Сначала сохраняем, потом показываем ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)

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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
        print(f"Сохранено: {save_path}")
    fig.show()


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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
        print(f"График сохранён: {save_path}")

    fig.show()