# Модуль multidomain.py

## Обзор

Модуль `multidomain.py` расширяет возможности библиотеки MPINN для решения задач теплопроводности в многослойных средах. Класс `MPINN` (Multi-domain PINN) позволяет моделировать системы с различными теплофизическими свойствами в разных слоях, автоматически обеспечивая непрерывность температуры и теплового потока на границах раздела сред.

## Зависимости

- `jax` — автоматическое дифференцирование и JIT-компиляция
- `flax.nnx` — нейронные сети
- `optax` — оптимизаторы
- `matplotlib` — визуализация
- `geom` — геометрические классы
- `pinn_core` — базовые классы PINN

---

## Класс MPINN

### Конструктор

```python
MPINN(nets, opt, weights, phys, n_collocation=100, rng=None)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `nets` | tuple[FCNet] | Нейронные сети (по одной на каждый домен) |
| `opt` | optax.GradientTransformation | Оптимизатор |
| `weights` | list[float] | Веса компонентов потерь |
| `phys` | object | Физические параметры |
| `n_collocation` | int | Количество точек коллокации на домен |
| `rng` | jax.Array | Ключ PRNG |

**Требуемые атрибуты `phys`:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `x0` | float | Левая граница области |
| `x1` | float | Правая граница области |
| `interfaces` | list[float] | Координаты границ между слоями |
| `all_lambdas` | list[float] | Теплопроводности каждого слоя |
| `T0` | float | Температура на левой границе |
| `T1` | float | Температура на правой границе (для Dirichlet) |

**Атрибуты экземпляра:**

- `boundaries` — кортеж всех границ (включая интерфейсы)
- `n_domains` — количество доменов (слоёв)
- `interfaces` — координаты интерфейсов
- `weights` — веса потерь
- `x_collocation` — точки коллокации для каждого домена
- `pinn_instances` — экземпляры PINN для каждого домена
- `graphdefs`, `params`, `txs`, `opt_states` — состояния моделей

---

### Метод `create_loss_fn`

```python
create_loss_fn(pde_fn, bc_left_fn, bc_right_fn, phys)
```

Создаёт функцию потерь для многослойной задачи.

**Компоненты потерь:**

1. **PDE losses** — невязка уравнения в каждом домене
2. **Boundary conditions** — условия на внешних границах
3. **Interface conditions** — условия сопряжения на границах слоёв:
   - Непрерывность температуры: `T0 = T1`
   - Непрерывность теплового потока: `λ_left·dT/dx0 = λ_right·dT/dx1`

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `pde_fn` | callable | Функция PDE |
| `bc_left_fn` | callable | Условие на левой границе |
| `bc_right_fn` | callable | Условие на правой границе |
| `phys` | object | Физические параметры |

**Возвращает:**

- `callable` — функция `total_loss(params_tuple, x_collocation)`

---

### Метод `train_step`

```python
train_step(params, x_collocation, txs, opt_states, loss_fn)
```

Один шаг обучения для всех доменов одновременно (JIT-компилируется).

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `params` | tuple | Параметры всех сетей |
| `x_collocation` | tuple | Точки коллокации для каждого домена |
| `txs` | tuple | Оптимизаторы |
| `opt_states` | tuple | Состояния оптимизаторов |
| `loss_fn` | callable | Функция потерь |

**Возвращает:**

- `new_params` — обновлённые параметры
- `new_opt_states` — новые состояния
- `total` — общие потери
- `aux` — детализированные потери по компонентам

---

### Метод `train_loop`

```python
train_loop(num_steps, loss_fn, loss_names, log_interval)
```

Цикл обучения.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `num_steps` | int | Количество шагов |
| `loss_fn` | callable | Функция потерь |
| `loss_names` | list[str] | Имена компонентов потерь |
| `log_interval` | int | Интервал логирования |

**Возвращает:**

- `dict` — история обучения

---

### Метод `fit`

```python
fit(pde_fn, bc_left_fn, bc_right_fn, phys, epochs, log_interval=100)
```

Полный цикл обучения многослойной модели.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `pde_fn` | callable | Функция PDE |
| `bc_left_fn` | callable | Левое граничное условие |
| `bc_right_fn` | callable | Правое граничное условие |
| `phys` | object | Физические параметры |
| `epochs` | int | Количество эпох |
| `log_interval` | int | Интервал логирования |

**Возвращает:**

- `history` — история обучения
- `training_time` — время обучения в секундах

**Имена компонентов потерь:**

```python
["pde_0", "pde_1", ..., "bc_left", "bc_right", "interface_0", "interface_1", ...]
```

---

### Метод `predict`

```python
predict(x_test)
```

Предсказание температуры в заданных точках с учётом доменов.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_test` | jax.Array | Координаты для предсказания |

**Возвращает:**

- `jax.Array` — предсказанные значения температуры

**Логика:**

Для каждой точки определяется, к какому домену она принадлежит, и используется соответствующая нейронная сеть.

---

### Метод `compute_metrics`

```python
compute_metrics(x_test, t_pred, t_exact)
```

Вычисление метрик качества.

**Возвращает:**

- `dict` — метрики: `mape`, `mae`, `mse`, `rmse`, `max_error`

---

### Метод `evaluate`

```python
evaluate(x_test, exact_fn, phys, bc_names=None)
```

Полная оценка модели.

**Возвращает:**

- `metrics` — словарь метрик
- `t_pred` — предсказанные значения
- `t_exact` — точные значения

---

### Метод `save_plot`

```python
save_plot(x_test, t_pred, t_exact, phys, save_path)
```

Сохранение графика с обозначением границ доменов.

**Особенности:**

- Вертикальные линии на позициях интерфейсов
- Сравнение предсказания и точного решения

---

### Метод `show_plot`

```python
show_plot(x_test, t_pred, t_exact, phys)
```

Отображение графика в интерактивной среде.

---

## Пример использования

### Двухслойная стена

```python
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from multidomain import MPINN
from pde import line_1d
from bc import dirichlet_bc


# Физические параметры для двухслойной стены
class PhysicsMulti:
    x0 = 0.0
    x1 = 0.3
    interfaces = [0.1]  # Граница между слоями на x = 0.1 м

    # Теплопроводности слоёв
    all_lambdas = [1.0, 0.5]  # Вт/(м·K)

    # Граничные условия
    T0 = 300.0  # K
    T1 = 400.0  # K


phys = PhysicsMulti()

# Создание нейронных сетей (по одной на домен)
rngs = nnx.Rngs(0)
nets = tuple(
    FCNet(din=1, dmid=50, dout=1, num_layers=4, activation=nnx.tanh, rngs=rngs)
    for _ in range(len(phys.all_lambdas))
)

# Веса потерь:
# [pde_domain_0, pde_domain_1, bc_left, bc_right, interface_0]
weights = [1.0, 1.0, 1.0, 1.0, 1.0]

# Создание MPINN
mpinn = MPINN(
    nets=nets, opt=optax.adam(1e-3), weights=weights, phys=phys, n_collocation=100
)

# Граничные условия
bc_left = lambda m: dirichlet_bc(m, phys.x0, phys.T0)
bc_right = lambda m: dirichlet_bc(m, phys.x1, phys.T1)

# Обучение
history, training_time = mpinn.fit(
    pde_fn=line_1d, bc_left_fn=bc_left, bc_right_fn=bc_right, phys=phys, epochs=10000
)

# Предсказание
x_test = jnp.linspace(phys.x0, phys.x1, 200).reshape(-1, 1)
T_pred = mpinn.predict(x_test)

# Оценка (если есть аналитическое решение)
from analytic import line_1d_dirichlet_exact

metrics, T_pred, T_exact = mpinn.evaluate(x_test, line_1d_dirichlet_exact, phys)
print(f"MAE: {metrics['mae']}, RMSE: {metrics['rmse']}")

# Сохранение графика
mpinn.save_plot(x_test, T_pred, T_exact, phys, "multilayer_solution.png")
```

### Трёхслойная стена

```python
class PhysicsThreeLayers:
    x0 = 0.0
    x1 = 0.3
    interfaces = [0.1, 0.2]  # Две границы раздела

    all_lambdas = [1.0, 0.5, 2.0]  # Три слоя
    T0 = 300.0
    T1 = 400.0


phys = PhysicsThreeLayers()

# Три нейронные сети
nets = tuple(FCNet(1, 50, 1, 4, nnx.tanh, nnx.Rngs(i)) for i in range(3))

# Веса: [pde×3, bc_left, bc_right, interface×2] = 7 компонентов
weights = [1.0] * 7

mpinn = MPINN(nets, optax.adam(1e-3), weights, phys)
```

---

## Структура весов потерь

Для N доменов структура весов следующая:

| Индекс | Компонент | Описание |
|--------|-----------|----------|
| `0 .. N-1` | `pde_i` | PDE в домене i |
| `N` | `bc_left` | Левое граничное условие |
| `N+1` | `bc_right` | Правое граничное условие |
| `N+2 .. N+2+(N-2)` | `interface_i` | Условия на интерфейсах |

**Пример для 2 доменов:**

```python
weights = [
    1.0,  # pde_0
    1.0,  # pde_1
    1.0,  # bc_left
    1.0,  # bc_right
    1.0,  # interface_0
]
```

**Пример для 3 доменов:**

```python
weights = [
    1.0,  # pde_0
    1.0,  # pde_1
    1.0,  # pde_2
    1.0,  # bc_left
    1.0,  # bc_right
    1.0,  # interface_0
    1.0,  # interface_1
]
```

---

## Условия на интерфейсах

На каждой границе раздела сред обеспечиваются два условия:

### 1. Непрерывность температуры

```
T0(x_interface) = T1(x_interface)
```

### 2. Непрерывность теплового потока

```
λ_left · dT/dx|_left = λ_right · dT/dx|_right
```

Эти условия добавляются в функцию потерь с соответствующими весами.

---

## Визуализация результатов

График, сохраняемый методом `save_plot`, включает:

- Синяя сплошная линия — аналитическое решение
- Красная пунктирная линия — решение PINN
- Серые вертикальные линии — позиции интерфейсов

---

## См. также

- [pinn_core.md](pinn_core.md) — Ядро PINN
- [pde.md](pde.md) — Дифференциальные уравнения
- [bc.md](bc.md) — Граничные условия
- [geom.md](geom.md) — Геометрия
