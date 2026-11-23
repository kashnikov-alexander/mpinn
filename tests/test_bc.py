"""Tests for boundary conditions module (extended, with nnx.Module models)."""

import jax
import jax.numpy as jnp
from flax import nnx

from mpinn.bc import dirichlet_bc, neumann_bc, robin_bc

# ===== Helper models (nnx.Module) =====


class LinearModel1D(nnx.Module):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a * x + self.b


class LinearModel2D(nnx.Module):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        return self.a * x[:, 0:1] + self.b * x[:, 1:2] + self.c


class QuadraticModel1D(nnx.Module):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        return self.a * x**2 + self.b * x + self.c


# ===== Tests =====


class TestDirichletBC:
    """Test cases for Dirichlet boundary conditions."""

    def test_dirichlet_analytical_1d(self):
        """Check exact zero residual for a model satisfying Dirichlet BC."""
        model = LinearModel1D(a=2.0, b=1.0)  # T(x) = 2x + 1
        points = jnp.array([[0.0], [0.5], [1.0]])
        values = 2.0 * points + 1.0  # exact values at those points
        residuals = dirichlet_bc(model, points, values)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_dirichlet_multiple_values(self):
        """Check residuals when values is an array."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0], [0.5], [1.0]])
        values = jnp.array([0.0, 0.5, 1.0])  # not matching model
        residuals = dirichlet_bc(model, points, values)
        expected = (2.0 * points + 1.0).ravel() - values
        assert jnp.allclose(residuals, expected, atol=1e-6)

    def test_dirichlet_2d(self):
        """Test Dirichlet works with 2D points."""
        model = LinearModel2D(a=1.0, b=2.0, c=0.0)  # T = x + 2y
        points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
        values = 1.0 * points[:, 0] + 2.0 * points[:, 1]
        residuals = dirichlet_bc(model, points, values)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_dirichlet_scalar_value_expanded(self):
        """Check scalar value broadcast to all points."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0], [0.5], [1.0]])
        scalar_value = 5.0
        residuals = dirichlet_bc(model, points, scalar_value)
        expected = (2.0 * points + 1.0).ravel() - 5.0
        assert jnp.allclose(residuals, expected, atol=1e-6)

    def test_dirichlet_jit(self):
        """Test JIT compilation compatibility."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0], [1.0]])
        values = jnp.array([1.0, 3.0])
        jit_fn = jax.jit(dirichlet_bc)
        residuals = jit_fn(model, points, values)
        expected = (2.0 * points + 1.0).ravel() - values
        assert jnp.allclose(residuals, expected, atol=1e-6)

    def test_dirichlet_output_shape(self, boundary_points, seed_key):
        """Check residuals shape."""
        x0 = boundary_points["left"]
        target = 1.0
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )
        residuals = dirichlet_bc(net, x0, target)
        assert residuals.shape == (x0.shape[0],)


class TestNeumanBC:
    """Test cases for Neumann boundary conditions."""

    def test_neumann_analytical_1d(self):
        """Exact zero residual when model derivative matches flux."""
        model = LinearModel1D(a=2.0, b=1.0)  # T' = 2
        points = jnp.array([[0.0], [0.5], [1.0]])
        normals = jnp.array([[1.0], [1.0], [1.0]])
        flux = 2.0
        residuals = neumann_bc(model, points, normals, flux)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_neumann_1d_nonzero(self):
        """Check residuals when derivative doesn't match."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0], [0.5], [1.0]])
        normals = jnp.array([[1.0], [1.0], [1.0]])
        flux = 3.0  # wrong flux
        residuals = neumann_bc(model, points, normals, flux)
        expected = 2.0 - 3.0
        assert jnp.allclose(residuals, expected, atol=1e-6)

    def test_neumann_with_negative_normal(self):
        """Check normal direction matters."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[1.0]])
        normals = jnp.array([[-1.0]])
        flux = -2.0  # derivative times normal = -2
        residuals = neumann_bc(model, points, normals, flux)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_neumann_2d(self):
        """Test Neumann with 2D points."""
        model = LinearModel2D(a=1.0, b=2.0, c=0.0)  # gradient = (1,2)
        points = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        normals = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        flux = jnp.array([1.0, 2.0])  # derivative in normal direction
        residuals = neumann_bc(model, points, normals, flux)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_neumann_jit(self):
        """Test JIT compilation."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0], [1.0]])
        normals = jnp.array([[1.0], [1.0]])
        flux = 2.0
        jit_fn = jax.jit(neumann_bc)
        residuals = jit_fn(model, points, normals, flux)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_neumann_vector_flux(self):
        """Check flux can be a vector (different for each point)."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0], [0.5], [1.0]])
        normals = jnp.array([[1.0], [1.0], [1.0]])
        flux = jnp.array([2.0, 2.0, 2.0])
        residuals = neumann_bc(model, points, normals, flux)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_neumann_quadratic_derivative(self):
        """Test with quadratic model (non-constant derivative)."""
        model = QuadraticModel1D(a=3.0, b=2.0, c=1.0)  # T' = 6x + 2
        x = jnp.array([[0.0], [0.5], [1.0]])
        normals = jnp.array([[1.0], [1.0], [1.0]])
        flux = 6.0 * x + 2.0  # exact derivative at each point
        residuals = neumann_bc(model, x, normals, flux)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)


class TestRobinBC:
    """Test cases for Robin boundary conditions."""

    def test_robin_analytical_1d(self):
        """Check zero residual for model satisfying Robin BC."""
        model = LinearModel1D(a=2.0, b=1.0)  # T = 2x + 1, T' = 2
        points = jnp.array([[0.0], [0.5], [1.0]])
        normals = jnp.array([[1.0], [1.0], [1.0]])
        alpha, beta = 3.0, 4.0
        T_vals = model(points).ravel()
        deriv = 2.0
        expected_value = alpha * T_vals + beta * deriv
        residuals = robin_bc(model, points, normals, alpha, beta, expected_value)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_robin_scalar_value_broadcast(self):
        """Check scalar value broadcast to all points."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0], [0.5], [1.0]])
        normals = jnp.array([[1.0], [1.0], [1.0]])
        alpha, beta = 1.0, 1.0
        T_vals = model(points).ravel()
        value = 5.0
        residuals = robin_bc(model, points, normals, alpha, beta, value)
        expected = T_vals + 2.0 - 5.0  # T' = 2
        assert jnp.allclose(residuals, expected, atol=1e-6)

    def test_robin_2d(self):
        """Test Robin with 2D points."""
        model = LinearModel2D(a=1.0, b=2.0, c=0.0)
        points = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        normals = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        alpha, beta = 2.0, 3.0
        T_vals = model(points).ravel()
        deriv = jnp.array([1.0, 2.0])
        value = alpha * T_vals + beta * deriv
        residuals = robin_bc(model, points, normals, alpha, beta, value)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_robin_jit(self):
        """Test JIT compilation."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0], [1.0]])
        normals = jnp.array([[1.0], [1.0]])
        alpha, beta = 1.0, 1.0
        value = 3.0
        jit_fn = jax.jit(robin_bc)
        residuals = jit_fn(model, points, normals, alpha, beta, value)
        assert jnp.all(jnp.isfinite(residuals))

    def test_robin_limits_neumann(self):
        """When alpha=0, Robin becomes Neumann: beta*dT/dn = value."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0]])
        normals = jnp.array([[1.0]])
        alpha = 0.0
        beta = 1.0
        value = 2.0
        residuals = robin_bc(model, points, normals, alpha, beta, value)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)

    def test_robin_limits_dirichlet(self):
        """When beta=0, Robin becomes Dirichlet: alpha*T = value."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.5]])
        normals = jnp.array([[1.0]])
        alpha = 1.0
        beta = 0.0
        value = model(points).ravel()[0]  # T at 0.5 = 2
        residuals = robin_bc(model, points, normals, alpha, beta, value)
        assert jnp.allclose(residuals, 0.0, atol=1e-6)


class TestBCEdgeCases:
    """Test edge cases for BC functions."""

    def test_dirichlet_empty_points(self):
        """Dirichlet with empty points should return empty array."""
        model = LinearModel1D(a=1.0, b=0.0)
        points = jnp.empty((0, 1))
        residuals = dirichlet_bc(model, points, 0.0)
        assert residuals.shape == (0,)

    def test_neumann_empty_points(self):
        """Neumann with empty points should return empty array."""
        model = LinearModel1D(a=1.0, b=0.0)
        points = jnp.empty((0, 1))
        normals = jnp.empty((0, 1))
        residuals = neumann_bc(model, points, normals, 0.0)
        assert residuals.shape == (0,)

    def test_robin_empty_points(self):
        """Robin with empty points should return empty array."""
        model = LinearModel1D(a=1.0, b=0.0)
        points = jnp.empty((0, 1))
        normals = jnp.empty((0, 1))
        residuals = robin_bc(model, points, normals, 1.0, 1.0, 0.0)
        assert residuals.shape == (0,)

    def test_neumann_normal_broadcasting(self):
        """Test that broadcasting works when normals shape is compatible."""
        model = LinearModel1D(a=2.0, b=1.0)
        points = jnp.array([[0.0], [1.0]])
        normals = jnp.array([[1.0]])  # shape (1,1) broadcasts to (2,1)
        flux = 2.0
        residuals = neumann_bc(model, points, normals, flux)
        # derivative = 2, normal=1, flux=2 => residual = 2*1 - 2 = 0 for both
        assert jnp.allclose(residuals, 0.0, atol=1e-6)
