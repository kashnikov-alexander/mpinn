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
from .bc import BC, AdiabaticBC, DirichletBC, NeumannBC, RobinBC
from .losses import loss_fn, loss_fn_split_bc
from .multidomain import (
    MPINN,
    init_opt_states,
    interface_loss,
    mpinn_loss_fn,
    predict,
)
from .pinn_core import (
    PINN,
    FCNet,
    ScaledNet,
    compute_metrics,
    evaluate,
    train_step,
)
from .plotting import (
    plot_2d,
    plot_2d_contour,
    plot_3d_isosurface,
    plot_3d_slices,
    plot_3d_surface_plotly,
    show_history,
    show_plot,
    show_plot_mpinn,
)
from .training import (
    fit,
    fit_adaptive,
    fit_adaptive_weights,
    init_opt_state,
    snapshot,
)
from .utils import save_results_to_csv

__all__ = [
    # bc
    "BC",
    # multidomain
    "MPINN",
    "PINN",
    "AdiabaticBC",
    "DirichletBC",
    # pinn_core
    "FCNet",
    "NeumannBC",
    "RobinBC",
    "ScaledNet",
    # модули
    "analytic",
    "bc",
    "compute_metrics",
    "evaluate",
    # training
    "fit",
    "fit_adaptive",
    "fit_adaptive_weights",
    "geom",
    "geometry_io",
    "init_opt_state",
    "init_opt_states",
    "interface_loss",
    # losses
    "loss_fn",
    "loss_fn_split_bc",
    "losses",
    "mpinn_loss_fn",
    "multidomain",
    "pde",
    "pinn_core",
    "plot_2d",
    "plot_2d_contour",
    "plot_3d_isosurface",
    "plot_3d_slices",
    "plot_3d_surface_plotly",
    "plotting",
    "predict",
    # utils
    "save_results_to_csv",
    # plotting
    "show_history",
    "show_plot",
    "show_plot_mpinn",
    "snapshot",
    "sources",
    "train_step",
    "training",
    "utils",
]
