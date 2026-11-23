# Модуль plotting.py

## Обзор

Модуль `plotting.py` предоставляет функции для визуализации результатов обучения PINN и MPINN моделей.

## Зависимости

- `matplotlib.pyplot` — построение графиков
- `jax.numpy` — численные операции
- `os` — работа с файловой системой

---

## Функции

### show_plot

```python
show_plot(x_test, T_pred, T_exact, phys, title="Сравнение PINN и точного решения")
```

Отображает график сравнения предсказания нейронной сети и точного аналитического решения.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_test` | jax.Array | Тестовые координаты (1D массив) |
| `T_pred` | jax.Array | Предсказанные значения температуры |
| `T_exact` | jax.Array | Точные значения температуры |
| `phys` | PhysicsParams | Объект с физическими параметрами (для границ) |
| `title` | str | Заголовок графика |

**Пример:**

```python
from mpinn.plotting import show_plot
from mpinn.pinn_core import PINN
import jax.numpy as jnp

# Предсказание обученной модели
x_test = jnp.linspace(0, 1, 100)
T_pred = pinn.predict(x_test)

# Точное решение
T_exact = exact_fn(x_test, phys)

# Показ графика
show_plot(x_test, T_pred, T_exact, phys, title="Решение задачи теплопроводности")
```

---

### save_plot

```python
save_plot(
    x_test, T_pred, T_exact, phys, save_path, title="Сравнение PINN и точного решения"
)
```

Сохраняет график сравнения предсказания и точного решения в файл.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x_test` | jax.Array | Тестовые координаты |
| `T_pred` | jax.Array | Предсказанные значения |
| `T_exact` | jax.Array | Точные значения |
| `phys` | PhysicsParams | Физические параметры |
| `save_path` | str | Путь для сохранения файла (например, 'results/solution.png') |
| `title` | str | Заголовок графика |

**Пример:**

```python
from mpinn.plotting import save_plot

save_plot(
    x_test,
    T_pred,
    T_exact,
    phys,
    save_path="results/pinn_solution.png",
    title="Сравнение PINN и аналитического решения",
)
```

---

### show_history

```python
show_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show_plot: bool = True
)
```

Отображает и/или сохраняет график истории обучения (потери по эпохам).

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `history` | Dict[str, List[float]] | Словарь истории обучения с ключами: `['steps', 'pde', 'bc_0', 'bc_1', 'total_loss']` |
| `save_path` | Optional[str] | Путь для сохранения графика (если None, не сохраняется) |
| `show_plot` | bool | Если True, отображает график на экране |

**Пример:**

```python
from mpinn.plotting import show_history

# После обучения
history, _ = pinn.fit(x_collocation, pde_fn, bc_fns, phys, epochs=10000)

# Показ истории обучения
show_history(history, save_path="results/training_history.png", show_plot=True)
```

---

## Стиль графиков

Все функции используют единый стиль визуализации:

### График решения (show_plot, save_plot)

- **Аналитическое решение**: синяя сплошная линия (`'b-'`, linewidth=2)
- **Решение PINN**: красная пунктирная линия (`'r:'`, linewidth=6)
- **Подписи осей**: 
  - X: "x, м"
  - Y: "T, К"
- **Легенда**: расположена в лучшем месте (`loc='best'`), размер шрифта 14
- **Сетка**: включена с прозрачностью 0.3
- **Границы**: устанавливаются по физическим параметрам (`phys.x0`, `phys.x1`)

### График истории (show_history)

- **PDE потери**: синяя линия
- **Граничные условия**: красные пунктирные линии (bc_0, bc_1)
- **Общая ошибка**: зелёная пунктирная линия
- **Масштаб Y**: логарифмический (`semilogy`)
- **Размер фигуры**: 12x7 дюймов

---

## Полный пример использования

```python
import jax.numpy as jnp
from mpinn.config import PhysicsParams, TrainConfig
from mpinn.pinn_core import FCNet, PINN
from mpinn.pde import line_1d
from mpinn.bc import dirichlet_bc
from mpinn.geom import Interval
from mpinn.analytic import line_1d_dirichlet_exact
from mpinn.plotting import show_plot, save_plot, show_history
import flax.nnx as nnx
import optax

# 1. Настройка
phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T1=400.0)

config = TrainConfig(
    hidden_features=64,
    num_layers=2,
    activation_name="GELU",
    lr=0.01,
    max_epochs=5000,
    patience=200,
    num_points=100,
)

# 2. Создание модели
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

# 3. Геометрия и точки коллокации
geom = Interval(phys.x0, phys.x1)
x_collocation = geom.generate_collocation(n_interior=config.num_points)

# 4. Граничные условия
bc_left = lambda m: dirichlet_bc(m, phys.x0, phys.T0)
bc_right = lambda m: dirichlet_bc(m, phys.x1, phys.T1)

# 5. Обучение
history, training_time = pinn.fit(
    x_collocation=x_collocation,
    pde_fn=line_1d,
    bc_fns=[bc_left, bc_right],
    phys=phys,
    epochs=config.max_epochs,
)

# 6. Предсказание и оценка
x_test = jnp.linspace(phys.x0, phys.x1, 100)
T_pred = pinn.predict(x_test)
T_exact = line_1d_dirichlet_exact(x_test, phys)

metrics, _, _ = pinn.evaluate(x_test, line_1d_dirichlet_exact, phys)
print(f"MSE: {metrics['mse']:.6e}, MAE: {metrics['mae']:.6e}")

# 7. Визуализация результатов
# Сохранение графика решения
save_plot(
    x_test,
    T_pred,
    T_exact,
    phys,
    save_path="results/solution.png",
    title="Решение задачи теплопроводности",
)

# Показ графика решения
show_plot(x_test, T_pred, T_exact, phys, title="PINN vs Аналитическое решение")

# Сохранение и показ истории обучения
show_history(history, save_path="results/training_history.png", show_plot=True)
```

---

## Визуализация для MPINN

Для многодоменных задач (MPINN) используйте те же функции, передавая предсказания от MPINN:

```python
from mpinn.multidomain import MPINN
from mpinn.plotting import show_plot, save_plot

# Обучение MPINN
mpinn = MPINN(nets, opt, phys, n_collocation=100)
history, _ = mpinn.fit(pde_fn, bc_left_fn, bc_right_fn, phys, epochs=5000)

# Предсказание
x_test = jnp.linspace(phys.x0, phys.x1, 200)
T_pred = mpinn.predict(x_test)
T_exact = exact_fn(x_test, phys)

# Визуализация (интерфейсы будут отмечены автоматически)
save_plot(
    x_test,
    T_pred,
    T_exact,
    phys,
    save_path="results/mpinn_solution.png",
    title="MPINN: Многослойная задача",
)
```

---

## Обработка путей к файлам

Функция `show_history` автоматически создаёт директории при сохранении:

```python
# Автоматическое создание директории results/plots/
show_history(history, save_path="results/plots/history.png")

# Проверка существования пути
import os

if not os.path.exists("results"):
    os.makedirs("results")

save_plot(x_test, T_pred, T_exact, phys, save_path="results/solution.png")
```

---

## Советы по визуализации

### Настройка качества изображения

```python
# Для презентаций (высокое DPI)
plt.savefig("solution_hd.png", dpi=300, bbox_inches="tight")

# Для веба (низкое DPI)
plt.savefig("solution_web.png", dpi=72, bbox_inches="tight")
```

### Кастомизация графиков

Для полной кастомизации используйте matplotlib напрямую:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_test, T_exact, "b-", label="Аналитическое", linewidth=2)
ax.plot(x_test, T_pred, "r--", label="PINN", linewidth=2)
ax.set_xlabel("x, м", fontsize=12)
ax.set_ylabel("T, К", fontsize=12)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)
plt.savefig("custom_plot.png", dpi=150)
plt.show()
```

---

## См. также

- [pinn_core.md](pinn_core.md) — Ядро PINN
- [multidomain.md](multidomain.md) — Многодоменные задачи
- [runner.md](runner.md) — Запуск экспериментов
