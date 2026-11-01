"""Tests for analytic solutions module (updated for factory pattern)."""

import jax.numpy as jnp
import pytest

from mpinn.analytic import (
    cylinder_1d_dirichlet_exact,
    cylinder_1d_neuman_exact,
    cylinder_1d_robin_exact,
    line_1d_dirichlet_exact,
    line_1d_neuman_exact,
    line_1d_robin_exact,
    sphere_1d_dirichlet_exact,
    sphere_1d_neuman_exact,
    sphere_1d_robin_exact,
)
from mpinn.utils import PhysicsParams


# Helper to create a dummy physics object with _lambda
@pytest.fixture
def phys():
    return PhysicsParams(_lambda=1.0, T_min=0.0, T_max=1000.0)


class TestAnalyticLine1D:
    """Test analytic solution for 1D line geometry."""

    def test_line_solution_shape(self, phys):
        """Check that solution has correct shape."""
        x = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
        exact_fn = line_1d_dirichlet_exact(T0=0.0, T1=1.0, x0=0.0, x1=1.0)
        u = exact_fn(x, phys)
        assert u.shape[0] == x.shape[0], f"Expected {x.shape[0]}, got {u.shape[0]}"

    def test_line_solution_boundary_values(self, phys):
        """Check that solution satisfies boundary conditions."""
        exact_fn = line_1d_dirichlet_exact(T0=5.0, T1=-3.0, x0=0.0, x1=1.0)
        x0 = jnp.array([[0.0]], dtype=jnp.float32)
        x1 = jnp.array([[1.0]], dtype=jnp.float32)

        u_left = exact_fn(x0, phys)
        u_right = exact_fn(x1, phys)

        assert jnp.isclose(u_left[0], 5.0, atol=1e-6), "Left BC not satisfied"
        assert jnp.isclose(u_right[0], -3.0, atol=1e-6), "Right BC not satisfied"

    def test_line_solution_dtype(self, phys):
        """Check dtype of analytic solution."""
        x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1).astype(jnp.float32)
        exact_fn = line_1d_dirichlet_exact(T0=0.0, T1=1.0, x0=0.0, x1=1.0)
        u = exact_fn(x, phys)
        assert u.dtype == jnp.float32

    def test_line_neumann_shape(self, phys):
        """Test Neuman solution shape."""
        x = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
        exact_fn = line_1d_neuman_exact(T0=300.0, flux=100.0, x0=0.0, x1=1.0)
        u = exact_fn(x, phys)
        assert u.shape[0] == x.shape[0]

    def test_line_robin_shape(self, phys):
        """Test Robin solution shape."""
        x = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
        exact_fn = line_1d_robin_exact(T0=300.0, T_inf=500.0, h=10.0, x0=0.0, x1=1.0)
        u = exact_fn(x, phys)
        assert u.shape[0] == x.shape[0]


class TestAnalyticCylinder1D:
    """Test analytic solution for 1D cylindrical geometry."""

    def test_cylinder_solution_shape(self, phys):
        """Check solution shape for cylindrical case."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)
        exact_fn = cylinder_1d_dirichlet_exact(T0=1.0, T1=0.0, r0=0.1, r1=1.0)
        u = exact_fn(x, phys)
        assert u.shape[0] == x.shape[0]

    def test_cylinder_solution_boundary_values(self, phys):
        """Check BCs for cylindrical solution."""
        r_inner, r_outer = 0.5, 2.0
        exact_fn = cylinder_1d_dirichlet_exact(T0=10.0, T1=0.0, r0=r_inner, r1=r_outer)

        x_inner = jnp.array([[r_inner]], dtype=jnp.float32)
        x_outer = jnp.array([[r_outer]], dtype=jnp.float32)

        u_inner = exact_fn(x_inner, phys)
        u_outer = exact_fn(x_outer, phys)

        assert jnp.isclose(u_inner[0], 10.0, rtol=1e-5)
        assert jnp.isclose(u_outer[0], 0.0, rtol=1e-5)

    def test_cylinder_neumann_shape(self, phys):
        """Test cylinder Neuman solution shape."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)
        exact_fn = cylinder_1d_neuman_exact(T0=300.0, flux=100.0, r0=0.1, r1=1.0)
        u = exact_fn(x, phys)
        assert u.shape[0] == x.shape[0]

    def test_cylinder_robin_shape(self, phys):
        """Test cylinder Robin solution shape."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)
        exact_fn = cylinder_1d_robin_exact(T0=300.0, T_inf=500.0, h=10.0, r0=0.1, r1=1.0)
        u = exact_fn(x, phys)
        assert u.shape[0] == x.shape[0]


class TestAnalyticSphere1D:
    """Test analytic solution for 1D spherical geometry."""

    def test_sphere_solution_shape(self, phys):
        """Check solution shape for spherical case."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)
        exact_fn = sphere_1d_dirichlet_exact(T0=1.0, T1=0.0, r0=0.1, r1=1.0)
        u = exact_fn(x, phys)
        assert u.shape[0] == x.shape[0]

    def test_sphere_solution_boundary_values(self, phys):
        """Check BCs for spherical solution."""
        r_inner, r_outer = 1.0, 3.0
        exact_fn = sphere_1d_dirichlet_exact(T0=100.0, T1=50.0, r0=r_inner, r1=r_outer)

        x_inner = jnp.array([[r_inner]], dtype=jnp.float32)
        x_outer = jnp.array([[r_outer]], dtype=jnp.float32)

        u_inner = exact_fn(x_inner, phys)
        u_outer = exact_fn(x_outer, phys)

        assert jnp.isclose(u_inner[0], 100.0, rtol=1e-5)
        assert jnp.isclose(u_outer[0], 50.0, rtol=1e-5)

    def test_sphere_neumann_shape(self, phys):
        """Test sphere Neuman solution shape."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)
        exact_fn = sphere_1d_neuman_exact(T0=300.0, flux=100.0, r0=0.1, r1=1.0)
        u = exact_fn(x, phys)
        assert u.shape[0] == x.shape[0]

    def test_sphere_robin_shape(self, phys):
        """Test sphere Robin solution shape."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)
        exact_fn = sphere_1d_robin_exact(T0=300.0, T_inf=500.0, h=10.0, r0=0.1, r1=1.0)
        u = exact_fn(x, phys)
        assert u.shape[0] == x.shape[0]


class TestAnalyticDeterminism:
    """Test determinism of analytic solutions."""

    def test_all_solutions_deterministic(self, phys):
        """All analytic solutions should be deterministic."""
        x = jnp.linspace(0.1, 1.0, 15).reshape(-1, 1)

        # Line Dirichlet
        exact_fn = line_1d_dirichlet_exact(T0=1.0, T1=0.0, x0=0.1, x1=1.0)
        u1 = exact_fn(x, phys)
        u2 = exact_fn(x, phys)
        assert jnp.allclose(u1, u2)

        # Cylinder Dirichlet
        exact_fn_cyl = cylinder_1d_dirichlet_exact(T0=1.0, T1=0.0, r0=0.1, r1=1.0)
        u1_cyl = exact_fn_cyl(x, phys)
        u2_cyl = exact_fn_cyl(x, phys)
        assert jnp.allclose(u1_cyl, u2_cyl)

        # Sphere Dirichlet
        exact_fn_sph = sphere_1d_dirichlet_exact(T0=1.0, T1=0.0, r0=0.1, r1=1.0)
        u1_sph = exact_fn_sph(x, phys)
        u2_sph = exact_fn_sph(x, phys)
        assert jnp.allclose(u1_sph, u2_sph)

        # Also test Neuman and Robin variants (use same parameters)
        exact_fn = line_1d_neuman_exact(T0=1.0, flux=0.0, x0=0.1, x1=1.0)
        u1 = exact_fn(x, phys)
        u2 = exact_fn(x, phys)
        assert jnp.allclose(u1, u2)

        exact_fn = line_1d_robin_exact(T0=1.0, T_inf=0.0, h=1.0, x0=0.1, x1=1.0)
        u1 = exact_fn(x, phys)
        u2 = exact_fn(x, phys)
        assert jnp.allclose(u1, u2)