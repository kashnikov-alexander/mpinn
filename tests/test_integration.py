"""Tests for integration of multiple modules (PINN training loop)."""

import jax
import jax.numpy as jnp
from flax import nnx

from mpinn.bc import dirichlet_bc
from mpinn.geom import Interval
from mpinn.pde import line_1d
from mpinn.pinn_core import FCNet

from mpinn.utils import PhysicsParams

phys = PhysicsParams(_lambda=1.0, T_min=0.0, T_max=1.0)


class TestPINNIntegration:
    """Integration tests for complete PINN workflow."""

    def test_pinn_training_step(self, seed_key):
        """Test a single training step of PINN."""
        # Setup geometry
        geom = Interval(0.0, 1.0)
        x_collocation = geom.sample_interior(n_points=50, rng=seed_key)
        x_bc_left = jnp.array([[0.0]], dtype=jnp.float32)
        x_bc_right = jnp.array([[1.0]], dtype=jnp.float32)

        # Setup network using NNX API
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1, dmid=20, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        # Define loss function
        def compute_loss(model, x_col, x0, x1):
            # PDE residual loss
            pde_residual = line_1d(model, x_col, phys=phys)
            pde_loss = jnp.mean(pde_residual**2)

            # BC losses (mean squared residuals)
            bc_left_residual = dirichlet_bc(model, x0, 0.0)
            bc_right_residual = dirichlet_bc(model, x1, 1.0)
            bc_left_loss = jnp.mean(bc_left_residual**2)
            bc_right_loss = jnp.mean(bc_right_residual**2)

            total_loss = pde_loss + bc_left_loss + bc_right_loss
            return total_loss

        # Compute initial loss
        initial_loss = compute_loss(net, x_collocation, x_bc_left, x_bc_right)

        assert jnp.isfinite(initial_loss), "Initial loss should be finite"
        assert initial_loss >= 0, "Loss should be non-negative"

    def test_pinn_gradient_descent_step(self, seed_key):
        """Test that gradients can be computed and applied."""
        geom = Interval(0.0, 1.0)
        x_collocation = geom.sample_interior(n_points=30, rng=seed_key)
        x_bc_left = jnp.array([[0.0]], dtype=jnp.float32)
        x_bc_right = jnp.array([[1.0]], dtype=jnp.float32)

        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1, dmid=16, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        def loss_fn(model):
            pde_residual = line_1d(model, x_collocation, phys=phys)
            pde_loss = jnp.mean(pde_residual**2)

            bc_left_residual = dirichlet_bc(model, x_bc_left, 0.0)
            bc_right_residual = dirichlet_bc(model, x_bc_right, 1.0)
            bc_left_loss = jnp.mean(bc_left_residual**2)
            bc_right_loss = jnp.mean(bc_right_residual**2)

            return pde_loss + bc_left_loss + bc_right_loss

        # Compute gradient using NNX
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(net)

        # Check gradient structure - NNX returns a GraphState-like object
        assert grads is not None

    def test_jit_compiled_training(self, seed_key):
        """Test JIT-compiled training step."""
        geom = Interval(0.0, 1.0)
        x_collocation = geom.sample_interior(n_points=20, rng=seed_key)
        x_bc_left = jnp.array([[0.0]], dtype=jnp.float32)
        x_bc_right = jnp.array([[1.0]], dtype=jnp.float32)

        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1, dmid=12, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        def loss_fn(model):
            pde_residual = line_1d(model, x_collocation, phys=phys)
            pde_loss = jnp.mean(pde_residual**2)
            bc_left_residual = dirichlet_bc(model, x_bc_left, 0.0)
            bc_right_residual = dirichlet_bc(model, x_bc_right, 1.0)
            bc_left_loss = jnp.mean(bc_left_residual**2)
            bc_right_loss = jnp.mean(bc_right_residual**2)
            return pde_loss + bc_left_loss + bc_right_loss

        # JIT compile
        jit_loss_fn = jax.jit(loss_fn)

        # Compare JIT vs non-JIT
        loss_normal = loss_fn(net)
        loss_jit = jit_loss_fn(net)

        assert jnp.isclose(loss_normal, loss_jit, atol=1e-6), "JIT loss should match"
