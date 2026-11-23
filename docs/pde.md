# Модуль pde.py

## Обзор

Модуль `pde.py` содержит реализации операторов дифференциальных уравнений в частных производных (PDE) для различных геометрий. Все функции вычисляют невязку уравнения в точках коллокации, которая затем минимизируется в процессе обучения PINN.

## Зависимости

- `jax` — автоматическое дифференцирование
- `jax.numpy` — численные операции

---

## Функции

### line_1d

Одномерное уравнение теплопроводности в декартовых координатах.

**Уравнение:**

```
d²T/dx² = 0
```

**Сигнатура:**

```python
line_1d(model, x, phys)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | nnx.Module | Нейронная сеть |
| `x` | jax.Array | Координаты формы `(n_points, 1)` |
| `phys` | object | Физические параметры (не используется в данном уравнении) |

**Возвращает:**

- `jax.Array` — среднее значение квадрата невязки уравнения

**Пример:**

```python
from pde import line_1d

# Вычисление невязки PDE
pde_loss = line_1d(model, x_collocation, phys)
```

---

### cylinder_1d

Одномерное уравнение теплопроводности в цилиндрических координатах (радиальное направление).

**Уравнение:**

```
(1/r) * dT/dr + d²T/dr² = 0
```

**Сигнатура:**

```python
cylinder_1d(model, x, phys)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | nnx.Module | Нейронная сеть |
| `x` | jax.Array | Радиальные координаты формы `(n_points, 1)` |
| `phys` | object | Физические параметры |

**Особенности:**

- Использует `r_safe` для избежания деления на ноль при r = 0
- Автоматическое дифференцирование через JAX для вычисления производных

**Пример:**

```python
from pde import cylinder_1d

# Для цилиндрической геометрии
pde_loss = cylinder_1d(model, r_points, phys)
```

---

### sphere_1d

Одномерное уравнение теплопроводности в сферических координатах (радиальное направление).

**Уравнение:**

```
d²T/dr² + (2/r) * dT/dr = 0
```

**Сигнатура:**

```python
sphere_1d(model, x, phys)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | nnx.Module | Нейронная сеть |
| `x` | jax.Array | Радиальные координаты формы `(n_points, 1)` |
| `phys` | object | Физические параметры |

**Особенности:**

- Использует `r_safe` для избежания деления на ноль при r = 0
- Множитель 2/r соответствует сферической симметрии

**Пример:**

```python
from pde import sphere_1d

# Для сферической геометрии
pde_loss = sphere_1d(model, r_points, phys)
```

---

## Математические детали

### Декартовы координаты (line_1d)

Для стационарной одномерной задачи теплопроводности без внутренних источников тепла:

```
d²T/dx² = 0
```

Общее решение: `T(x) = C₁·x + C₂`

### Цилиндрические координаты (cylinder_1d)

Для радиального направления в цилиндре:

```
(1/r) · d/dr(r · dT/dr) = 0
```

Раскрытая форма:

```
(1/r) · dT/dr + d²T/dr² = 0
```

Общее решение: `T(r) = C₁·ln(r) + C₂`

### Сферические координаты (sphere_1d)

Для радиального направления в сфере:

```
(1/r²) · d/dr(r² · dT/dr) = 0
```

Раскрытая форма:

```
d²T/dr² + (2/r) · dT/dr = 0
```

Общее решение: `T(r) = C₁/r + C₂`

---

## Использование в PINN

Функции PDE используются для вычисления компонента потерь, отвечающего за выполнение дифференциального уравнения во внутренних точках области:

```python
from pinn_core import PINN
from pde import line_1d

# Создание функции потерь
loss_fn = pinn.create_loss_fn(
    pde_fn=line_1d, bc_left=bc_left_fn, bc_right=bc_right_fn, phys=phys
)

# В процессе обучения вычисляется:
# total_loss = w_pde * pde_loss + w_bc1 * bc1_loss + w_bc2 * bc2_loss
```

---

## Пример полного использования

```python
import jax.numpy as jnp
import flax.nnx as nnx
from pinn_core import FCNet, PINN
from pde import line_1d, cylinder_1d, sphere_1d
from bc import dirichlet_bc
from geom import Interval
import optax


# Физические параметры для разных геометрий
class PhysicsLine:
    x0 = 0.0
    x1 = 1.0
    T0 = 300.0
    T1 = 400.0


class PhysicsCylinder:
    x0 = 0.1  # внутренний радиус
    x1 = 0.5  # внешний радиус
    T0 = 300.0
    T1 = 400.0


class PhysicsSphere:
    x0 = 0.1
    x1 = 0.5
    T0 = 300.0
    T1 = 400.0


# Выбор геометрии и соответствующего PDE
geometry_type = "cylinder"  # "line", "cylinder", или "sphere"

if geometry_type == "line":
    phys = PhysicsLine()
    pde_fn = line_1d
elif geometry_type == "cylinder":
    phys = PhysicsCylinder()
    pde_fn = cylinder_1d
else:
    phys = PhysicsSphere()
    pde_fn = sphere_1d

# Создание и обучение модели
net = FCNet(1, 50, 1, 4, nnx.tanh, nnx.Rngs(0))
pinn = PINN(net, optax.adam(1e-3), weights=[1.0, 1.0, 1.0])

geom = Interval(phys.x0, phys.x1)
x_collocation = geom.generate_collocation(n_interior=100)

bc_left = lambda m: dirichlet_bc(m, phys.x0, phys.T0)
bc_right = lambda m: dirichlet_bc(m, phys.x1, phys.T1)

history, time = pinn.fit(
    x_collocation=x_collocation,
    pde_fn=pde_fn,
    bc_fns=[bc_left, bc_right],
    phys=phys,
    epochs=10000,
)
```

---

## См. также

- [pinn_core.md](pinn_core.md) — Ядро PINN
- [bc.md](bc.md) — Граничные условия
- [analytic.md](analytic.md) — Аналитические решения
- [geom.md](geom.md) — Геометрия
