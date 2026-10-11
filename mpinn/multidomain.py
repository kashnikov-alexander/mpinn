"""
MPINN: мультидоменная физически-информированная нейронная сеть.

Архитектура: координатор (MPINN) управляет списком независимых экземпляров PINN.
Каждый PINN инкапсулирует свою сеть, оптимизатор, физику и граничные условия.
Обучение совместное (один train_step), обновление параметров раздельное.
"""

import time
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from functools import partial
from .pinn_core import PINN


class MPINN:
    def __init__(
        self,
        domain_configs: list[dict],
        interface_pairs: list[tuple],        # [(idx1, name1, idx2, name2), ...]
        n_interface_points: int,
        interface_weight: float = 1.0,
        pde_weight: float = 1.0,
        bc_weight:  float = 1.0,
        rng: jax.Array | None = None,
    ):
        """
        Инициализация MPINN.

        Args:
            domain_configs: Список конфигураций доменов. Каждая конфигурация:
                {
                    'geom':       GeometryBase,           # Геометрия домена
                    'net':        nnx.Module,             # Сеть домена
                    'opt':        optax.GradientTransformation,
                    'pde_fn':     Callable,
                    'bc_configs': list[dict],             # ГУ на ВНЕШНИХ границах
                    'n_points':   int,                    # Точек коллокации
                    'weights':    tuple[float, float],    # (w_pde, w_bc)
                    'lam':        float = 1.0,            # Теплопроводность для интерфейса
                }
            interface_pairs: Список кортежей (idx1, name1, idx2, name2).
            n_interface_points: Количество точек на каждом интерфейсе.
            interface_weight: Вес потерь на интерфейсах.
            rng: Ключ JAX.
        """
        self.n_domains = len(domain_configs)
        self.domain_configs = domain_configs
        self.interface_pairs = interface_pairs
        self.interface_weight = interface_weight
        self.pde_weight = float(pde_weight)
        self.bc_weight  = float(bc_weight)

        if rng is None:
            rng = jax.random.PRNGKey(0)

        # --- 1. Список PINN, по одному на домен ---
        # Каждый PINN инкапсулирует свои graphdef, params, tx, opt_state, loss_fn
        self.pinns = [
            PINN(
                net=cfg['net'],
                opt=cfg['opt'],
                weights=(1.0, 1.0),
                pde_fn=cfg['pde_fn'],
                bc_configs=cfg['bc_configs'],
                lam=cfg['lam'],
                source_fn=cfg.get('source_fn', 0.0),
            )
            for cfg in domain_configs
        ]

        # --- 2. Точки коллокации ---
        col_keys = jax.random.split(rng, self.n_domains)
        self.x_collocation = tuple(
            cfg['geom'].sample_interior(cfg['n_points'], rng=k)
            for cfg, k in zip(domain_configs, col_keys)
        )

        # --- 3. Интерфейсные точки ---
        itf_keys = jax.random.split(rng, len(interface_pairs))
        self.interface_data = []            # (pts, n1, n2, idx1, idx2)
        for (idx1, name1, idx2, name2), k in zip(interface_pairs, itf_keys):
            geom1 = domain_configs[idx1]['geom']
            geom2 = domain_configs[idx2]['geom']
            pts, n1, n2 = geom1.sample_interface(
                other_geom=geom2,
                n_points=n_interface_points,
                self_interface_name=name1,
                other_interface_name=name2,
                rng=k,
            )
            self.interface_data.append((pts, n1, n2, idx1, idx2))

        # --- 4. lambda по доменам (для потока на интерфейсе) ---
        self.lambdas = tuple(cfg.get('lam', 1.0) for cfg in domain_configs)

        # --- 5. Общий loss_fn создаётся один раз ---
        self.loss_fn = self.create_loss_fn()

    # ------------------------------------------------------------------ loss

    def create_loss_fn(self):
        """Собирает общий лосс: PDE + BC всех доменов + интерфейсные условия."""
        pinns = self.pinns
        x_collocs = self.x_collocation
        itf_data = self.interface_data
        lambdas = self.lambdas
        itf_w   = self.interface_weight
        w_pde   = self.pde_weight
        w_bc    = self.bc_weight

        def total_loss(params_tuple):
            models = tuple(
                nnx.merge(p.graphdef, p_params)
                for p, p_params in zip(pinns, params_tuple)
            )

            # --- PDE + BC по доменам ---
            pde_total = jnp.array(0.0)
            bc_total  = jnp.array(0.0)
            for pinn, model, x_d in zip(pinns, models, x_collocs):
                _, aux = pinn.loss_fn(model, x_d)          # aux = (pde, *bcs)
                pde_total = pde_total + aux[0]
                if len(aux) > 1:
                    bc_total = bc_total + sum(aux[1:])

            # --- Интерфейсные лоссы ---
            itf_total = jnp.array(0.0)
            for pts, n1, n2, idx1, idx2 in itf_data:
                m1, m2 = models[idx1], models[idx2]
                lam1, lam2 = lambdas[idx1], lambdas[idx2]

                T1 = m1(pts).ravel()
                T2 = m2(pts).ravel()
                cont_T = jnp.mean((T1 - T2) ** 2)

                def grad_normal(model, points, normals):
                    def predict_one(x):
                        return model(x.reshape(1, -1)).ravel()[0]
                    g = jax.vmap(jax.grad(predict_one))(points)
                    return jnp.sum(g * normals, axis=1)

                dT1n = grad_normal(m1, pts, n1)
                dT2n = grad_normal(m2, pts, n2)
                cont_flux = jnp.mean((lam1 * dT1n + lam2 * dT2n) ** 2)

                itf_total = itf_total + cont_T + cont_flux

            # --- Итоговый лосс (формула из статьи) ---
            total = w_pde * pde_total + w_bc * bc_total + itf_w * itf_total

            return total, (pde_total, bc_total, itf_total)

        return total_loss

    # --------------------------------------------------------------- training

    @partial(jax.jit, static_argnames=['self', 'loss_fn'])
    def train_step(self, params_list, opt_states, loss_fn):
        """
        Один шаг обучения.
        params_list — кортеж params (по одному на PINN)
        opt_states — кортеж opt_state (по одному на PINN)
        """
        def closure(pl):
            return loss_fn(pl)

        (total, aux), grads_list = jax.value_and_grad(closure, has_aux=True)(params_list)

        new_params = []
        new_opt_states = []
        for p, g, tx, os in zip(params_list, grads_list, self.txs, opt_states):
            updates, new_os = tx.update(g, os)
            new_params.append(optax.apply_updates(p, updates))
            new_opt_states.append(new_os)

        return tuple(new_params), tuple(new_opt_states), total, aux

    @property
    def txs(self):
        """Оптимизаторы всех PINN."""
        return tuple(p.tx for p in self.pinns)

    @property
    def params(self):
        """Текущие params всех PINN."""
        return tuple(p.params for p in self.pinns)

    @property
    def opt_states(self):
        return tuple(p.opt_state for p in self.pinns)

    def fit(self, epochs: int = 1000, log_interval: int = 100):
        loss_fn = self.loss_fn
        loss_names = ("pde_total", "bc_total", "itf_total")

        history = {"total_loss": []}
        for name in loss_names:
            history[name] = []

        curr_params = self.params
        curr_opt_states = self.opt_states

        t0 = time.perf_counter()
        for step in range(epochs):
            curr_params, curr_opt_states, total, aux = self.train_step(
                curr_params, curr_opt_states, loss_fn
            )
            if step % log_interval == 0 or step == epochs - 1:
                history["total_loss"].append(float(total))
                for name, val in zip(loss_names, aux):
                    history[name].append(float(val))

        # --- Записываем params/opt_state обратно в PINN ---
        for pinn, p, os in zip(self.pinns, curr_params, curr_opt_states):
            pinn.params = p
            pinn.opt_state = os

        return history, time.perf_counter() - t0

    # --------------------------------------------------------------- predict

    def predict(self, x_test):
        """Предсказание по всем доменам с учетом их геометрии (1D/2D/3D)."""
        x_flat = jnp.atleast_1d(x_test)
        dim = self.domain_configs[0]['geom'].dim
        if x_flat.ndim == 1:
            x_flat = x_flat.reshape(-1, dim)

        t_pred = jnp.full((x_flat.shape[0],), jnp.nan)

        for i, pinn in enumerate(self.pinns):
            geom = self.domain_configs[i]['geom']
            model = nnx.merge(pinn.graphdef, pinn.params)

            if hasattr(geom, 'is_inside'):
                mask = geom.is_inside(x_flat)
            else:
                bmin, bmax = geom.bounds
                mask = jnp.all((x_flat >= bmin) & (x_flat <= bmax), axis=1)

            # Не перезаписываем точки, уже отнесенные к другим доменам
            already = ~jnp.isnan(t_pred)
            mask = mask & (~already)

            if jnp.any(mask):
                preds = model(x_flat[mask]).ravel()
                t_pred = t_pred.at[mask].set(preds)

        return t_pred.reshape(-1, 1)

    # --------------------------------------------------------------- helpers

    def evaluate(self, x_test, exact_fn):
        t_pred = self.predict(x_test).ravel()
        t_exact = exact_fn(x_test).ravel()
        diff = t_pred - t_exact
        return {
            "mse":  float(jnp.mean(diff ** 2)),
            "mae":  float(jnp.mean(jnp.abs(diff))),
            "rmse": float(jnp.sqrt(jnp.mean(diff ** 2))),
            "max_err":  float(jnp.max(jnp.abs(diff))),
        }, t_pred, t_exact

    def _make_rl2_evaluator(self, x_test, exact_fn):
        graphdefs = tuple(p.graphdef for p in self.pinns)
        domain_configs = self.domain_configs
        dim = domain_configs[0]['geom'].dim

        @jax.jit
        def _rl2(params_list):
            x_flat = jnp.atleast_1d(x_test)
            if x_flat.ndim == 1:
                x_flat = x_flat.reshape(-1, dim)

            t_pred = jnp.full((x_flat.shape[0],), jnp.nan)
            for i, (g, p) in enumerate(zip(graphdefs, params_list)):
                model = nnx.merge(g, p)
                geom = domain_configs[i]['geom']
                if hasattr(geom, 'is_inside'):
                    mask = geom.is_inside(x_flat)
                else:
                    bmin, bmax = geom.bounds
                    mask = jnp.all((x_flat >= bmin) & (x_flat <= bmax), axis=1)
                already = ~jnp.isnan(t_pred)
                mask = mask & (~already)
                preds = model(x_flat).ravel()
                t_pred = jnp.where(mask, preds, t_pred)

            t_exact = exact_fn(x_test).ravel()
            return jnp.linalg.norm(t_pred - t_exact) / (jnp.linalg.norm(t_exact) + 1e-12)

        return _rl2

    def _snapshot_params(self):
        return tuple(jax.tree.map(jnp.copy, p) for p in self.params)

    def _restore_params(self, snapshot):
        for pinn, p in zip(self.pinns, snapshot):
            pinn.params = p

    def fit_adaptive(
        self,
        x_test,
        exact_fn,
        target_rl2: float = 1e-3,
        block_epochs: int = 500,
        max_epochs: int = 50_000,
        log_interval: int = 500,
        patience: int = 5,
        tol: float = 1e-6,
    ):
        """
        Адаптивное обучение MPINN блоками до достижения целевого RL2.

        Parameters
        ----------
        x_test : array (M, D)
            Тестовые точки, по которым оценивается RL2.
        exact_fn : callable
            Функция точного решения, принимающая x_test.
        target_rl2 : float
            Целевое значение относительной L2-ошибки.
        block_epochs : int
            Длина одного блока обучения в эпохах.
        max_epochs : int
            Максимальное число эпох.
        log_interval : int
            Частота логирования внутри блока.
        patience : int
            Число блоков без улучшения до остановки.
        tol : float
            Минимальное улучшение, считающееся прогрессом.

        Returns
        -------
        history : dict
            История обучения, склеенная по блокам.
        total_time : float
            Полное время обучения в секундах.
        """
        import time

        eval_fn = self._make_rl2_evaluator(x_test, exact_fn)

        history = {}
        total_epochs = 0
        best_rl2 = float('inf')
        best_params = self._snapshot_params()
        best_epoch = 0
        no_improve = 0

        t0 = time.perf_counter()

        while total_epochs < max_epochs:
            block = min(block_epochs, max_epochs - total_epochs)

            block_history, _ = self.fit(epochs=block, log_interval=log_interval)
            total_epochs += block

            for key, values in block_history.items():
                history.setdefault(key, []).extend(values)

            rl2 = float(eval_fn(self.params))

            if rl2 < best_rl2 - tol:
                best_rl2 = rl2
                best_params = self._snapshot_params()
                best_epoch = total_epochs
                no_improve = 0
            else:
                no_improve += 1

            pde_last = block_history.get("pde_total", [float("nan")])[-1]
            bc_last  = block_history.get("bc_total",  [float("nan")])[-1]
            if_last  = block_history.get("itf_total", [float("nan")])[-1]

            print(
                f"[block {total_epochs:6d}] "
                f"RL2={rl2:.4e}  "
                f"PDE={pde_last:.3e}  BC={bc_last:.3e}  IF={if_last:.3e}  "
                f"(best RL2={best_rl2:.4e} @ {best_epoch}, "
                f"patience={no_improve}/{patience})"
)

            if rl2 < target_rl2:
                print(f"Target RL2 = {target_rl2:.2e} достигнут на эпохе {total_epochs}")
                break

            if no_improve >= patience:
                print(f"Ранняя остановка: нет прогресса за {patience} блоков")
                break

        # восстановление лучших параметров
        self._restore_params(best_params)

        total_time = time.perf_counter() - t0
        return history, total_time