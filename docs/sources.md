# Модуль sources.py

## Обзор

Модуль `sources.py` предоставляет функции для создания источников тепла в стационарных задачах теплопроводности. Источники представляются как функции f(x), где x — пространственная координата.

Уравнение теплопроводности с источником:

```
-d/dx(k·dT/dx) = f(x)
```

где:
- `T` — температура
- `k` — теплопроводность
- `f(x)` — функция источника тепла (Вт/м³)

## Зависимости

- `jax.numpy` — численные операции
- `typing.Callable` — аннотации типов

---

## Функции

### constant_source

```python
constant_source(value: float) -> Callable[[Array], Array]
```

Создаёт функцию постоянного источника тепла f(x) = value.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `value` | float | Значение константы (Вт/м³) |

**Возвращает:**

- `Callable[[Array], Array]` — функция источника, принимающая x и возвращающая f(x)

**Пример:**

```python
from mpinn.sources import constant_source

# Постоянный источник 1000 Вт/м³
source_fn = constant_source(1000.0)

# Использование в PhysicsParams
phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T1=400.0, source_fn=source_fn)
```

---

### gaussian_source

```python
gaussian_source(
    center: float,
    width: float,
    amplitude: float
) -> Callable[[Array], Array]
```

Создаёт функцию гауссова источника тепла.

Формула:
```
f(x) = amplitude · exp(-(x - center)² / (2 · width²))
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `center` | float | Центр источника (м) |
| `width` | float | Ширина (sigma) источника (м) |
| `amplitude` | float | Амплитуда источника (Вт/м³) |

**Возвращает:**

- `Callable[[Array], Array]` — функция источника

**Пример:**

```python
from mpinn.sources import gaussian_source

# Гауссов источник с центром в 0.5 м, шириной 0.1 м, амплитудой 5000 Вт/м³
source_fn = gaussian_source(center=0.5, width=0.1, amplitude=5000.0)

phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T1=400.0, source_fn=source_fn)
```

---

### point_source

```python
point_source(
    location: float,
    epsilon: float = 0.01,
    amplitude: float = 1.0
) -> Callable[[Array], Array]
```

Аппроксимирует точечный источник (дельта-функцию) узким гауссианом.

Формула:
```
f(x) = norm · exp(-(x - location)² / (2 · epsilon²))
```

где `norm = amplitude / (√(2π) · epsilon)` для нормировки интеграла к amplitude.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `location` | float | Положение точечного источника (м) |
| `epsilon` | float | Ширина аппроксимации (чем меньше, тем ближе к дельте) |
| `amplitude` | float | Интегральная мощность источника (Вт) |

**Возвращает:**

- `Callable[[Array], Array]` — функция источника

**Пример:**

```python
from mpinn.sources import point_source

# Точечный источник в точке x=0.3 с мощностью 100 Вт
source_fn = point_source(location=0.3, epsilon=0.01, amplitude=100.0)

phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T1=400.0, source_fn=source_fn)
```

---

### custom_source

```python
custom_source(func: Callable[[Array], Array]) -> Callable[[Array], Array]
```

Обёртка для пользовательской функции источника.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `func` | Callable[[Array], Array] | Пользовательская функция f(x) |

**Возвращает:**

- `Callable[[Array], Array]` — та же функция

**Пример:**

```python
from mpinn.sources import custom_source
import jax.numpy as jnp


# Линейно возрастающий источник
def linear_source(x):
    return 1000.0 * x  # f(x) = 1000·x


source_fn = custom_source(linear_source)


# Или синусоидальный источник
def sinusoidal_source(x):
    return 500.0 * jnp.sin(2 * jnp.pi * x)


source_fn = custom_source(sinusoidal_source)

phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T1=400.0, source_fn=source_fn)
```

---

### no_source

```python
no_source() -> Callable[[Array], Array]
```

Возвращает функцию нулевого источника (однородное уравнение).

**Возвращает:**

- `Callable[[Array], Array]` — функция, всегда возвращающая 0

**Пример:**

```python
from mpinn.sources import no_source

# Явное указание отсутствия источника
phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T1=400.0, source_fn=no_source())

# Эквивалентно (по умолчанию source_fn=None)
phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T1=400.0)
```

---

## Полный пример использования

```python
import jax.numpy as jnp
from mpinn.config import PhysicsParams, TrainConfig
from mpinn.pinn_core import FCNet, PINN
from mpinn.pde import line_1d
from mpinn.bc import dirichlet_bc
from mpinn.geom import Interval
from mpinn.sources import constant_source, gaussian_source, point_source, custom_source
import flax.nnx as nnx
import optax

# === Пример 1: Постоянный источник ===

phys1 = PhysicsParams(
    x0=0.0,
    x1=1.0,
    T0=300.0,
    T1=400.0,
    source_fn=constant_source(1000.0),  # 1000 Вт/м³
)

# === Пример 2: Гауссов источник ===

phys2 = PhysicsParams(
    x0=0.0,
    x1=1.0,
    T0=300.0,
    T1=400.0,
    source_fn=gaussian_source(center=0.5, width=0.1, amplitude=5000.0),
)

# === Пример 3: Точечный источник ===

phys3 = PhysicsParams(
    x0=0.0,
    x1=1.0,
    T0=300.0,
    T1=400.0,
    source_fn=point_source(location=0.3, epsilon=0.01, amplitude=100.0),
)

# === Пример 4: Пользовательский источник ===


def polynomial_source(x):
    """Квадратичный источник f(x) = 1000·(1 - x²)"""
    return 1000.0 * (1.0 - x**2)


phys4 = PhysicsParams(
    x0=0.0, x1=1.0, T0=300.0, T1=400.0, source_fn=custom_source(polynomial_source)
)

# === Обучение модели с источником ===

config = TrainConfig(
    hidden_features=64,
    num_layers=2,
    activation_name="GELU",
    lr=0.01,
    max_epochs=5000,
    patience=200,
    num_points=150,  # Больше точек для лучшего разрешения источника
)

net = FCNet(
    din=1,
    dmid=config.hidden_features,
    dout=1,
    num_layers=config.num_layers,
    activation=nnx.gelu,
    rngs=nnx.Rngs(0),
)

optimizer = optax.adam(config.lr)
pinn = PINN(net, opt=optimizer, weights=config.weights)

geom = Interval(phys1.x0, phys1.x1)
x_collocation = geom.generate_collocation(n_interior=config.num_points)

bc_left = lambda m: dirichlet_bc(m, phys1.x0, phys1.T0)
bc_right = lambda m: dirichlet_bc(m, phys1.x1, phys1.T1)

history, training_time = pinn.fit(
    x_collocation=x_collocation,
    pde_fn=line_1d,  # line_1d автоматически учитывает source_fn из phys
    bc_fns=[bc_left, bc_right],
    phys=phys1,
    epochs=config.max_epochs,
)

print(f"Обучение завершено за {training_time:.2f}c")
print(f"PDE потеря: {history['pde'][-1]:.6e}")
```

---

## Влияние источников на PDE

Модуль `pde.py` автоматически учитывает источник при вычислении невязки:

```python
# Из pde.py:line_1d
def line_1d(model, x, phys):
    # ... вычисление второй производной ...

    # Вычисляем источник если задан
    source_val = 0.0
    if hasattr(phys, "source_fn") and phys.source_fn is not None:
        source_val = phys.source_fn(x_flat)

    # Невязка: d²T/dx² + f(x) = 0
    residual = d2T_dx2 + source_val

    return jnp.mean(residual**2)
```

---

## Рекомендации по выбору параметров

### Для constant_source

- Типичные значения: 100 – 10000 Вт/м³
- Подходит для моделирования равномерного тепловыделения

### Для gaussian_source

- `center`: должен находиться внутри области [x0, x1]
- `width`: обычно 0.01 – 0.2 м (зависит от размера области)
- `amplitude`: 1000 – 10000 Вт/м³

### Для point_source

- `epsilon`: 0.001 – 0.05 м (меньше = ближе к дельта-функции)
- `amplitude`: полная мощность в Вт (не плотность!)
- Требует больше точек коллокации для точного разрешения

---

## Советы по использованию

### Увеличение количества точек коллокации

Для задач с источниками рекомендуется увеличить количество точек:

```python
# Без источника
config = TrainConfig(num_points=100)

# С источником (особенно точечным)
config = TrainConfig(num_points=200)
```

### Визуализация источника

```python
import matplotlib.pyplot as plt
import jax.numpy as jnp

x = jnp.linspace(0, 1, 200)
source_fn = gaussian_source(center=0.5, width=0.1, amplitude=5000.0)
f_values = source_fn(x)

plt.figure(figsize=(8, 4))
plt.plot(x, f_values, "r-", linewidth=2)
plt.xlabel("x, м")
plt.ylabel("f(x), Вт/м³")
plt.title("Распределение источника тепла")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## См. также

- [pde.md](pde.md) — Дифференциальные уравнения
- [config.md](config.md) — Физические параметры
- [pinn_core.md](pinn_core.md) — Ядро PINN
