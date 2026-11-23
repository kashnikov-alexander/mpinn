# Модуль pinn_core.py

## Обзор

Модуль `pinn_core.py` является ядром библиотеки MPINN. Он содержит основные классы для построения полносвязных нейронных сетей и реализации метода PINN (Physics-Informed Neural Networks).

## Зависимости

- `jax` — автоматическое дифференцирование и JIT-компиляция
- `flax.nnx` — функциональный API для нейронных сетей
- `optax` — оптимизаторы
- `matplotlib` — визуализация
- `jax.numpy` — численные операции

---

## Классы

### FCNet

Полносвязная нейронная сеть (Fully Connected Network) для аппроксимации решения PDE.

#### Конструктор

```python
FCNet(din, dmid, dout, num_layers, activation, rngs)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `din` | int | Размерность входного слоя (обычно 1 для 1D задач) |
| `dmid` | int | Размерность скрытых слоёв |
| `dout` | int | Размерность выходного слоя (обычно 1) |
| `num_layers` | int | Количество скрытых слоёв |
| `activation` | callable | Функция активации (например, `nnx.tanh`) |
| `rngs` | nnx.Rngs | Генератор случайных чисел для инициализации весов |

**Атрибуты:**

- `layers` — список скрытых слоёв `nnx.Linear`
- `linear_out` — выходной слой
- `num_layers` — количество скрытых слоёв
- `activation` — функция активации

#### Метод `__call__`

```python
__call__(x)
```

Прямой проход через сеть.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x` | jax.Array | Входные данные формы `(batch_size, din)` |

**Возвращает:**

- `jax.Array` — выходные данные формы `(batch_size, dout)`

**Пример:**

```python
import flax.nnx as nnx
from pinn_core import FCNet

rngs = nnx.Rngs(42)
net = FCNet(din=1, dmid=50, dout=1, num_layers=4, activation=nnx.tanh, rngs=rngs)

x = jnp.array([[0.5]])
y = net(x)  # Прямой проход
```

---

### PINN

Класс для обучения и использования физически информированных нейронных сетей.

#### Конструктор

```python
PINN(net, opt, weights)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `net` | FCNet | Нейронная сеть |
| `opt` | optax.GradientTransformation | Оптимизатор (например, `optax.adam()`) |
| `weights` | list[float] | Веса для компонентов функции потерь [PDE, BC1, BC2, ...] |

**Атрибуты:**

- `net` — нейронная сеть
- `graphdef` — структура графа сети (для JIT)
- `params` — параметры сети
- `tx` — оптимизатор
- `opt_state` — состояние оптимизатора
- `weights` — веса компонентов потерь

#### Метод `create_loss_fn`

```python
create_loss_fn(pde_fn, *bc_fns, phys)
```

Создаёт функцию потерь для обучения.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `pde_fn` | callable | Функция PDE (из модуля `pde.py`) |
| `*bc_fns` | callable | Функции граничных условий (из модуля `bc.py`) |
| `phys` | object | Объект с физическими параметрами |

**Возвращает:**

- `callable` — функция потерь `total_loss(model, x_collocation)`

#### Метод `train_step`

```python
train_step(params, graphdef, x_collocation, tx, opt_state, loss_fn)
```

Один шаг градиентного спуска (JIT-компилируется).

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `params` | dict | Параметры модели |
| `graphdef` | GraphDef | Структура графа |
| `x_collocation` | jax.Array | Точки коллокации |
| `tx` | optax.GradientTransformation | Оптимизатор |
| `opt_state` | OptState | Состояние оптимизатора |
| `loss_fn` | callable | Функция потерь |

**Возвращает:**

- `new_params` — обновлённые параметры
- `new_opt_state` — новое состояние оптимизатора
- `total_loss` — общее значение потерь
- `aux_losses` — кортеж значений потерь по компонентам

#### Метод `train_loop`

```python
train_loop(x_collocation, num_steps, loss_fn, loss_names, log_interval=100)
```

Цикл обучения модели.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_collocation` | jax.Array | Точки коллокации |
| `num_steps` | int | Количество шагов обучения |
| `loss_fn` | callable | Функция потерь |
| `loss_names` | list[str] | Имена компонентов потерь |
| `log_interval` | int | Интервал логирования |

**Возвращает:**

- `dict` — история обучения:
  - `steps` — номера шагов
  - `total_loss` — значения общих потерь
  - `{name}` — значения каждого компонента потерь

#### Метод `fit`

```python
fit(x_collocation, pde_fn, bc_fns, phys, epochs)
```

Полный цикл обучения модели.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_collocation` | jax.Array | Точки коллокации |
| `pde_fn` | callable | Функция PDE |
| `bc_fns` | list[callable] | Список функций граничных условий |
| `phys` | object | Физические параметры |
| `epochs` | int | Количество эпох обучения |

**Возвращает:**

- `history` — история обучения
- `training_time` — время обучения в секундах

**Пример:**

```python
from pinn_core import PINN, FCNet
from pde import line_1d
from bc import dirichlet_bc
import optax

net = FCNet(1, 50, 1, 4, nnx.tanh, nnx.Rngs(0))
pinn = PINN(net, optax.adam(1e-3), weights=[1.0, 1.0, 1.0])

bc_left = lambda m: dirichlet_bc(m, 0.0, 300.0)
bc_right = lambda m: dirichlet_bc(m, 1.0, 400.0)

history, time = pinn.fit(
    x_collocation=x_points,
    pde_fn=line_1d,
    bc_fns=[bc_left, bc_right],
    phys=phys,
    epochs=10000,
)
```

#### Метод `predict`

```python
predict(x_test)
```

Предсказание решения в заданных точках.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_test` | jax.Array | Точки для предсказания |

**Возвращает:**

- `jax.Array` — предсказанные значения

#### Метод `compute_metrics`

```python
compute_metrics(x_test, T_pred, T_exact)
```

Вычисление метрик качества.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_test` | jax.Array | Точки |
| `T_pred` | jax.Array | Предсказанные значения |
| `T_exact` | jax.Array | Точные значения |

**Возвращает:**

- `dict` — метрики:
  - `mape` — средняя абсолютная процентная ошибка
  - `mae` — средняя абсолютная ошибка
  - `mse` — среднеквадратичная ошибка
  - `rmse` — корень из MSE
  - `max_error` — максимальная ошибка

#### Метод `evaluate`

```python
evaluate(x_test, exact_fn, phys, bc_names=None)
```

Полная оценка модели с вычислением метрик.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_test` | jax.Array | Точки для оценки |
| `exact_fn` | callable | Функция точного решения |
| `phys` | object | Физические параметры |
| `bc_names` | list[str] | Названия граничных условий |

**Возвращает:**

- `metrics` — словарь метрик
- `T_pred` — предсказанные значения
- `T_exact` — точные значения

#### Метод `save_plot`

```python
save_plot(x_test, T_pred, T_exact, phys, save_path)
```

Сохранение графика сравнения предсказания и точного решения.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_test` | jax.Array | Координаты |
| `T_pred` | jax.Array | Предсказание |
| `T_exact` | jax.Array | Точное решение |
| `phys` | object | Физические параметры |
| `save_path` | str | Путь для сохранения файла |

#### Метод `show_plot`

```python
show_plot(x_test, T_pred, T_exact, phys)
```

Отображение графика сравнения (в интерактивной среде).

---

## Функции

### normalize

```python
normalize(data, min_val, max_val)
```

Нормализация данных к диапазону [0, 1].

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `data` | jax.Array/float | Исходные данные |
| `min_val` | float | Минимум диапазона |
| `max_val` | float | Максимум диапазона |

**Возвращает:**

- Нормализованные данные в диапазоне [0, 1]

### denormalize

```python
denormalize(data_norm, min_val, max_val)
```

Денормализация данных из [0, 1] обратно в исходный диапазон.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `data_norm` | jax.Array/float | Нормализованные данные |
| `min_val` | float | Минимум исходного диапазона |
| `max_val` | float | Максимум исходного диапазона |

**Возвращает:**

- Восстановленные физические данные

---

## Пример полного использования

```python
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from pinn_core import FCNet, PINN
from pde import line_1d
from bc import dirichlet_bc
from geom import Interval


# Физические параметры
class Physics:
    x0 = 0.0
    x1 = 1.0
    T0 = 300.0
    T1 = 400.0


phys = Physics()

# Создание сети
rngs = nnx.Rngs(0)
net = FCNet(din=1, dmid=50, dout=1, num_layers=4, activation=nnx.tanh, rngs=rngs)

# PINN
optimizer = optax.adam(1e-3)
weights = [1.0, 1.0, 1.0]
pinn = PINN(net, optimizer, weights)

# Геометрия
geometry = Interval(phys.x0, phys.x1)
x_collocation = geometry.generate_collocation(n_interior=100)

# Граничные условия
bc_left = lambda m: dirichlet_bc(m, phys.x0, phys.T0)
bc_right = lambda m: dirichlet_bc(m, phys.x1, phys.T1)

# Обучение
history, training_time = pinn.fit(
    x_collocation=x_collocation,
    pde_fn=line_1d,
    bc_fns=[bc_left, bc_right],
    phys=phys,
    epochs=10000,
)

# Предсказание и оценка
x_test = jnp.linspace(0, 1, 100).reshape(-1, 1)
T_pred = pinn.predict(x_test)

from analytic import line_1d_dirichlet_exact

T_exact = line_1d_dirichlet_exact(x_test.ravel(), phys)

metrics, _, _ = pinn.evaluate(x_test, line_1d_dirichlet_exact, phys)
print(f"MAE: {metrics['mae']}, RMSE: {metrics['rmse']}")

# Сохранение результата
pinn.save_plot(x_test, T_pred, T_exact, phys, "solution.png")
```

---

## См. также

- [pde.md](pde.md) — Дифференциальные уравнения
- [bc.md](bc.md) — Граничные условия
- [analytic.md](analytic.md) — Аналитические решения
- [geom.md](geom.md) — Геометрия
- [multidomain.md](multidomain.md) — Многослойные задачи
