# Модуль geom.py

## Обзор

Модуль `geom.py` предоставляет классы для определения геометрических областей и генерации точек коллокации, необходимых для обучения PINN. Точки коллокации используются для вычисления невязки дифференциального уравнения во внутренних точках области и на границах.

## Зависимости

- `jax` — генерация случайных чисел
- `jax.numpy` — численные операции
- `abc` — абстрактные базовые классы

---

## Классы

### Geometry (абстрактный класс)

Базовый абстрактный класс для всех геометрий.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `dim` | int | Размерность пространства (1 или 2) |

**Абстрактные методы:**

```python
sample_interior(self, n_points, method="random", rng=None)
```
Генерирует точки внутри области.

```python
sample_boundary(self)
```
Генерирует точки на границе области.

---

### Interval (одномерная геометрия)

Одномерный интервал [x0, x1].

#### Конструктор

```python
Interval(x0, x1)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x0` | float | Левая граница интервала |
| `x1` | float | Правая граница интервала |

**Атрибуты:**

- `dim = 1`
- `x0` — левая граница
- `x1` — правая граница

#### Метод `sample_interior`

```python
sample_interior(n_points, method="random", rng=None)
```

Генерирует точки внутри интервала.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `n_points` | int | Количество точек |
| `method` | str | `'random'` или `'uniform'` |
| `rng` | jax.Array | Ключ PRNG (опционально) |

**Возвращает:**

- `jax.Array` формы `(n_points, 1)`

**Пример:**

```python
from geom import Interval

interval = Interval(0.0, 1.0)

# Случайные точки
x_random = interval.sample_interior(100, method="random")

# Равномерная сетка
x_uniform = interval.sample_interior(100, method="uniform")
```

#### Метод `sample_boundary`

```python
sample_boundary()
```

Возвращает граничные точки (левую и правую).

**Возвращает:**

- `jax.Array` формы `(2, 1)` — [[x0], [x1]]

**Пример:**

```python
boundary = interval.sample_boundary()
# array([[0.], [1.]])
```

#### Метод `generate_collocation`

```python
generate_collocation(n_interior=100, method="random", rng=None)
```

Генерирует полный набор точек для обучения: внутренние + граничные.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `n_interior` | int | Количество внутренних точек |
| `method` | str | Метод генерации внутренних точек |
| `rng` | jax.Array | Ключ PRNG |

**Возвращает:**

- `jax.Array` формы `(n_interior + 2, 1)`

**Пример:**

```python
x_collocation = interval.generate_collocation(n_interior=100)
# 100 внутренних + 2 граничные = 102 точки
```

---

### Rectangle (двумерная геометрия)

Двумерный прямоугольник [x_min, x_max] × [y_min, y_max].

#### Конструктор

```python
Rectangle(x_min, x_max, y_min, y_max)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_min` | float | Минимум по оси X |
| `x_max` | float | Максимум по оси X |
| `y_min` | float | Минимум по оси Y |
| `y_max` | float | Максимум по оси Y |

**Атрибуты:**

- `dim = 2`
- `x_min`, `x_max` — границы по X
- `y_min`, `y_max` — границы по Y

#### Метод `sample_interior`

```python
sample_interior(n_points, method="random", rng=None)
```

Генерирует точки внутри прямоугольника.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `n_points` | int | Количество точек |
| `method` | str | `'random'` или `'uniform'` |
| `rng` | jax.Array | Ключ PRNG |

**Возвращает:**

- `jax.Array` формы `(n_points, 2)` — координаты (x, y)

**Пример:**

```python
from geom import Rectangle

rect = Rectangle(0.0, 1.0, 0.0, 0.5)

# Случайные точки
points = rect.sample_interior(200, method="random")

# Равномерная сетка
grid = rect.sample_interior(200, method="uniform")
```

#### Метод `sample_boundary`

```python
sample_boundary(n_points=None, method="random", rng=None)
```

Генерирует точки на периметре прямоугольника.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `n_points` | int | Количество точек (если None, возвращает 4 угла) |
| `method` | str | `'random'` — равномерное распределение по периметру |
| `rng` | jax.Array | Ключ PRNG |

**Возвращает:**

- `jax.Array` формы `(n_points, 2)` или `(4, 2)` если n_points=None

**Пример:**

```python
# Четыре угла
corners = rect.sample_boundary()
# array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

# 100 случайных точек на периметре
boundary = rect.sample_boundary(n_points=100)
```

#### Метод `generate_collocation`

```python
generate_collocation(
    n_interior=100,
    n_boundary=None,
    method_interior="random",
    method_boundary="random",
    rng=None,
)
```

Генерирует полный набор точек: внутренние + граничные.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `n_interior` | int | Количество внутренних точек |
| `n_boundary` | int | Количество граничных точек (None = только углы) |
| `method_interior` | str | Метод для внутренних точек |
| `method_boundary` | str | Метод для граничных точек |
| `rng` | jax.Array | Ключ PRNG |

**Возвращает:**

- `jax.Array` формы `(n_interior + n_boundary, 2)`

**Пример:**

```python
x_collocation = rect.generate_collocation(
    n_interior=200, n_boundary=50, method_interior="random", method_boundary="random"
)
```

---

## Использование в PINN

### Одномерная задача

```python
from pinn_core import PINN
from geom import Interval

# Геометрия
phys_x0 = 0.0
phys_x1 = 1.0
geometry = Interval(phys_x0, phys_x1)

# Точки коллокации
x_collocation = geometry.generate_collocation(n_interior=100)

# Обучение
pinn.fit(x_collocation=x_collocation, ...)
```

### Двумерная задача

```python
from geom import Rectangle

# Прямоугольная область 1м × 0.5м
rect = Rectangle(0.0, 1.0, 0.0, 0.5)

# Точки для обучения
x_collocation = rect.generate_collocation(n_interior=200, n_boundary=50)
```

---

## Методы генерации точек

### Random (случайная выборка)

Точки генерируются равномерно случайным образом в области.

**Преимущества:**
- Хорошее покрытие области при большом количестве точек
- Подходит для стохастической оптимизации

**Недостатки:**
- Возможны кластеры точек

### Uniform (равномерная сетка)

Точки распределяются равномерно по области.

**Преимущества:**
- Детерминированное расположение
- Равномерное покрытие

**Недостатки:**
- Может быть менее эффективно для некоторых задач

---

## Воспроизводимость результатов

Для воспроизводимости используйте явный ключ PRNG:

```python
import jax.random as random

key = random.PRNGKey(42)
geometry = Interval(0.0, 1.0)

# Генерация с фиксированным seed
x_points = geometry.sample_interior(100, method="random", rng=key)
```

---

## Пример полного использования

```python
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from pinn_core import FCNet, PINN
from pde import line_1d
from bc import dirichlet_bc
from analytic import line_1d_dirichlet_exact
from geom import Interval


# Физические параметры
class Physics:
    x0 = 0.0
    x1 = 1.0
    T0 = 300.0
    T1 = 400.0


phys = Physics()

# Создание геометрии
geometry = Interval(phys.x0, phys.x1)

# Генерация точек коллокации
x_collocation = geometry.generate_collocation(n_interior=100, method="random")

print(f"Форма точек коллокации: {x_collocation.shape}")
# (102, 1) - 100 внутренних + 2 граничные

# Создание и обучение модели
net = FCNet(1, 50, 1, 4, nnx.tanh, nnx.Rngs(0))
pinn = PINN(net, optax.adam(1e-3), weights=[1.0, 1.0, 1.0])

bc_left = lambda m: dirichlet_bc(m, phys.x0, phys.T0)
bc_right = lambda m: dirichlet_bc(m, phys.x1, phys.T1)

history, time = pinn.fit(
    x_collocation=x_collocation,
    pde_fn=line_1d,
    bc_fns=[bc_left, bc_right],
    phys=phys,
    epochs=10000,
)

# Оценка
x_test = jnp.linspace(phys.x0, phys.x1, 100).reshape(-1, 1)
metrics, T_pred, T_exact = pinn.evaluate(x_test, line_1d_dirichlet_exact, phys)
print(f"MAE: {metrics['mae']}")
```

---

## См. также

- [pinn_core.md](pinn_core.md) — Ядро PINN
- [pde.md](pde.md) — Дифференциальные уравнения
- [bc.md](bc.md) — Граничные условия
