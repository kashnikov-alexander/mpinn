# Модуль config.py

## Обзор

Модуль `config.py` предоставляет классы конфигурации для определения физических параметров задачи и гиперпараметров обучения модели PINN.

## Зависимости

- `dataclasses` — для объявления классов данных
- `optax` — оптимизаторы
- `flax.nnx` — функции активации
- `jax.numpy` — численные операции

---

## Классы

### PhysicsParams

Класс для хранения физических параметров задачи теплопроводности.

#### Атрибуты

| Атрибут | Тип | Описание | Значение по умолчанию |
|---------|-----|----------|----------------------|
| `x0` | float | Левая граница области, м | None |
| `x1` | float | Правая граница области, м | None |
| `T0` | float | Температура на левой границе, К | None |
| `T_inf` | float | Температура окружающей среды, К | None |
| `_lambda` | float | Теплопроводность, Вт/(м·К) | None |
| `h` | float | Коэффициент теплоотдачи, Вт/(м²·К) | None |
| `source_fn` | Callable | Функция источника тепла f(x) | None |

#### Свойства

##### alpha_right

```python
@property
def alpha_right(self):
    return self.h
```

Коэффициент при T в условии Робина: α = h

##### beta

```python
@property
def beta(self):
    return self._lambda
```

Коэффициент при dT/dx в условии Робина: β = λ

##### beta

```python
@property
def beta(self):
    return self.h * self.T_inf
```

Свободный член в условии Робина: γ = h · T_inf

#### Пример использования

```python
from mpinn.config import PhysicsParams

# Создание объекта с параметрами по умолчанию
phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T_inf=500.0, _lambda=1.0, h=10.0)

# Доступ к свойствам условия Робина
alpha = phys.alpha_right  # 10.0
beta = phys.beta  # 1.0
gamma = phys.gamma  # 5000.0
```

---

### TrainConfig

Класс конфигурации обучения модели, включающий архитектуру сети, оптимизатор и параметры обучения.

#### Атрибуты

| Атрибут | Тип | Описание | Значение по умолчанию |
|---------|-----|----------|----------------------|
| `hidden_features` | int | Количество нейронов в скрытых слоях | 64 |
| `num_layers` | int | Количество скрытых слоёв | 2 |
| `activation_name` | str | Название функции активации | 'GELU' |
| `opt_name` | str | Название оптимизатора | 'adam' |
| `lr` | float | Скорость обучения | 0.01 |
| `max_epochs` | int | Максимальное количество эпох | 3000 |
| `patience` | int | Число эпох без улучшения до Early Stopping | 200 |
| `min_delta` | float | Минимальное изменение для учета как улучшения | 1e-6 |
| `monitor` | str | Метрика для мониторинга Early Stopping | 'total_loss' |
| `num_points` | int | Количество точек коллокации | 100 |
| `weights` | Tuple[float, float, float] | Веса потерь [pde, bc_0, bc_1] | (1.0, 1.0, 1.0) |
| `save_img` | bool | Сохранять ли график решения | False |
| `show_plot` | bool | Показывать ли график решения | False |
| `image_path` | Optional[str] | Путь для сохранения графика | None |

#### Пример использования

```python
from mpinn.config import TrainConfig

# Конфигурация по умолчанию
config = TrainConfig()

# Кастомная конфигурация
config = TrainConfig(
    hidden_features=128,
    num_layers=4,
    activation_name="tanh",
    opt_name="adam",
    lr=0.001,
    max_epochs=5000,
    patience=300,
    min_delta=1e-7,
    num_points=200,
    weights=(1.0, 2.0, 2.0),
    save_img=True,
    image_path="results/solution.png",
)
```

---

## Функции фабрики

### get_activation

```python
get_activation(name: str)
```

Фабрика функций активации. Возвращает функцию активации по названию.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `name` | str | Название активации: 'tanh', 'Swish', 'sin', 'GELU', 'ReLU', 'Sigmoid' |

**Возвращает:**

- callable — функция активации из `flax.nnx` или `jax.numpy`

**Пример:**

```python
from mpinn.config import get_activation
import flax.nnx as nnx

tanh_fn = get_activation("tanh")  # nnx.tanh
gelu_fn = get_activation("GELU")  # nnx.gelu
relu_fn = get_activation("ReLU")  # nnx.relu
sin_fn = get_activation("sin")  # jnp.sin
```

---

### get_optimizer

```python
get_optimizer(name: str, lr: float)
```

Фабрика оптимизаторов. Возвращает объект оптимизатора optax.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `name` | str | Название оптимизатора: 'adam', 'sgd', 'adagrad', 'rmsprop' |
| `lr` | float | Скорость обучения |

**Возвращает:**

- optax.GradientTransformation — объект оптимизатора

**Пример:**

```python
from mpinn.config import get_optimizer

adam_opt = get_optimizer("adam", lr=0.001)
sgd_opt = get_optimizer("sgd", lr=0.01)
rmsprop_opt = get_optimizer("rmsprop", lr=0.005)
```

---

## Константы по умолчанию

### DEFAULT_PHYSICS

Словарь физических параметров по умолчанию:

```python
DEFAULT_PHYSICS = {
    "x0": 0.0,
    "x1": 1.0,
    "T0": 300.0,
    "T_inf": 500.0,
    "_lambda": 1.0,
    "h": 10.0,
}
```

### DEFAULT_TRAIN_CONFIG

Словарь параметров обучения по умолчанию:

```python
DEFAULT_TRAIN_CONFIG = {
    "hidden_features": 64,
    "num_layers": 2,
    "activation_name": "GELU",
    "opt_name": "adam",
    "lr": 0.01,
    "max_epochs": 3000,
    "patience": 200,
    "min_delta": 1e-6,
    "num_points": 100,
    "weights": (1.0, 1.0, 1.0),
}
```

---

## Полный пример использования

```python
from mpinn.config import PhysicsParams, TrainConfig, get_activation, get_optimizer
from mpinn.pinn_core import FCNet, PINN
from mpinn.geom import Interval
import flax.nnx as nnx

# 1. Физические параметры
phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T_inf=500.0, _lambda=1.0, h=10.0)

# 2. Конфигурация обучения
config = TrainConfig(
    hidden_features=64,
    num_layers=2,
    activation_name="GELU",
    opt_name="adam",
    lr=0.01,
    max_epochs=3000,
    patience=200,
    num_points=100,
    weights=(1.0, 1.0, 1.0),
)

# 3. Создание модели
act_fn = get_activation(config.activation_name)
optimizer = get_optimizer(config.opt_name, config.lr)

net = FCNet(
    din=1,
    dmid=config.hidden_features,
    dout=1,
    num_layers=config.num_layers,
    activation=act_fn,
    rngs=nnx.Rngs(0),
)

pinn = PINN(net, opt=optimizer, weights=config.weights)

# 4. Геометрия и точки коллокации
geom = Interval(phys.x0, phys.x1)
x_collocation = geom.generate_collocation(n_interior=config.num_points)

print(f"Точки коллокации: {x_collocation.shape}")
print(f"Границы: [{phys.x0}, {phys.x1}]")
print(f"Условие Робина: α={phys.alpha_right}, β={phys.beta}, γ={phys.gamma}")
```

---

## См. также

- [pinn_core.md](pinn_core.md) — Ядро PINN
- [pde.md](pde.md) — Дифференциальные уравнения
- [bc.md](bc.md) — Граничные условия
- [runner.md](runner.md) — Запуск экспериментов
