"""mpinn: PINN на JAX + Equinox, 1D/2D/3D, многодоменная (MPINN)."""

# --- модули (для from mpinn import <module>) ---
from . import (
    analytic,
    bc,
    geom,
    geometry_io,
    losses,
    multidomain,
    pde,
    pinn_core,
    plotting,
    sources,
    training,
    utils,
)

# --- ключевые символы (для from mpinn import <symbol>) ---
from .bc import BC, DirichletBC, NeumannBC, RobinBC, AdiabaticBC
from .pinn_core import (
    FCNet, ScaledNet, PINN, train_step, evaluate, compute_metrics,
)
from .losses import loss_fn, loss_fn_split_bc
from .training import (
    fit, fit_adaptive, fit_adaptive_weights, init_opt_state, snapshot,
)
from .multidomain import (
    MPINN, init_opt_states, predict,
    mpinn_loss_fn, interface_loss,
)
from .plotting import (
    show_history, show_plot_mpinn, show_plot, save_plot,
    plot_2d_contour, plot_2d,
    plot_3d_slices, plot_3d_isosurface, plot_3d_surface_plotly,
)
from .utils import save_results_to_csv

__all__ = [
    # модули
    "analytic", "bc", "geom", "geometry_io", "losses", "multidomain",
    "pde", "pinn_core", "plotting", "sources", "training", "utils",
    # bc
    "BC", "DirichletBC", "NeumannBC", "RobinBC", "AdiabaticBC",
    # pinn_core
    "FCNet", "ScaledNet", "PINN", "train_step", "evaluate", "compute_metrics",
    # losses
    "loss_fn", "loss_fn_split_bc",
    # training
    "fit", "fit_adaptive", "fit_adaptive_weights", "init_opt_state", "snapshot",
    # multidomain
    "MPINN", "init_opt_states", "predict", "mpinn_loss_fn", "interface_loss",
    # plotting
    "show_history", "show_plot_mpinn", "show_plot", "save_plot",
    "plot_2d_contour", "plot_2d",
    "plot_3d_slices", "plot_3d_isosurface", "plot_3d_surface_plotly",
    # utils
    "save_results_to_csv",
]