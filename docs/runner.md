# Модуль runner.py

## Обзор

Модуль `runner.py` предоставляет высокоуровневый API для запуска экспериментов с PINN, включая обучение с Early Stopping и Grid Search гиперпараметров.

## Зависимости

- `pandas` — обработка результатов
- `matplotlib` — визуализация
- `jax.numpy` — численные операции
- `functools.partial` — частичное применение функций

---

## Функции

### run_experiment

```python
run_experiment(
    config: TrainConfig,
    phys: Optional[PhysicsParams] = None,
    pde_fn=None,
    exact_fn=None,
    bc_fns_override: Optional[List] = None
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]
```

Запуск одного эксперимента с механизмом Early Stopping.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `config` | TrainConfig | Конфигурация обучения |
| `phys` | PhysicsParams | Физические параметры (если None, используются значения по умолчанию) |
| `pde_fn` | callable | Функция PDE (если None, используется `line_1d`) |
| `exact_fn` | callable | Функция точного решения для верификации |
| `bc_fns_override` | List[callable] | Список функций граничных условий (если None, создаются автоматически) |

**Возвращает:**

- `metrics` — словарь с метриками качества (mse, mape, mae, rmse, max_error)
- `history` — история обучения (losses по эпохам)

**Пример:**

```python
from mpinn.runner import run_experiment
from mpinn.config import TrainConfig, PhysicsParams

# Конфигурация
config = TrainConfig(
    hidden_features=64,
    num_layers=2,
    activation_name="GELU",
    lr=0.01,
    max_epochs=3000,
    patience=200,
    num_points=100,
    weights=(1.0, 1.0, 1.0),
    save_img=True,
    image_path="results/solution.png",
)

phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T_inf=500.0, _lambda=1.0, h=10.0)

# Запуск эксперимента
metrics, history = run_experiment(config=config, phys=phys)

print(f"MSE: {metrics['mse']}, MAE: {metrics['mae']}")
print(f"Обучено эпох: {metrics['epochs_trained']}, Время: {metrics['training_time']}c")
```

---

### run_grid_search

```python
run_grid_search(
    param_grid: Dict[str, List[Any]],
    base_config: Optional[TrainConfig] = None,
    phys: Optional[PhysicsParams] = None,
    csv_path: str = 'csv_results/1D_line_robin_results.csv',
    pde_fn=None,
    exact_fn=None
) -> pd.DataFrame
```

Grid Search: автоматический перебор комбинаций гиперпараметров.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `param_grid` | Dict[str, List] | Словарь {имя_параметра: [список_значений]} |
| `base_config` | TrainConfig | Базовая конфигурация (если None, создается новая) |
| `phys` | PhysicsParams | Физические параметры |
| `csv_path` | str | Путь для сохранения результатов CSV |
| `pde_fn` | callable | Функция PDE |
| `exact_fn` | callable | Функция точного решения |

**Возвращает:**

- `pd.DataFrame` — DataFrame с результатами всех экспериментов

**Пример:**

```python
from mpinn.runner import run_grid_search
from mpinn.config import TrainConfig, PhysicsParams

# Сетка параметров для перебора
param_grid = {
    "hidden_features": [32, 64, 128],
    "num_layers": [2, 3, 4],
    "activation_name": ["tanh", "GELU"],
    "lr": [0.001, 0.01, 0.1],
}

# Базовая конфигурация
base_config = TrainConfig(max_epochs=3000, patience=200, num_points=100)

phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T_inf=500.0, _lambda=1.0, h=10.0)

# Запуск Grid Search
df_results = run_grid_search(
    param_grid=param_grid,
    base_config=base_config,
    phys=phys,
    csv_path="results/grid_search.csv",
)

# Анализ результатов
print(f"Всего экспериментов: {len(df_results)}")
print(f"Лучший MSE: {df_results['mse'].min()}")
print(f"Лучшая конфигурация:")
best = df_results.loc[df_results["mse"].idxmin()]
print(best[["layers", "neurons", "activation_func", "lr", "mse"]])
```

---

### _train_with_early_stopping (внутренняя)

```python
_train_with_early_stopping(
    pinn: PINN,
    x_collocation: jnp.ndarray,
    pde_fn,
    bc_fns: List,
    phys: PhysicsParams,
    max_epochs: int,
    patience: int,
    min_delta: float,
    monitor: str = 'total_loss'
) -> Tuple[Dict[str, List[float]], float]
```

Обучение модели с механизмом Early Stopping.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `pinn` | PINN | Объект PINN |
| `x_collocation` | jnp.ndarray | Точки коллокации |
| `pde_fn` | callable | Функция PDE |
| `bc_fns` | List[callable] | Список функций граничных условий |
| `phys` | PhysicsParams | Физические параметры |
| `max_epochs` | int | Максимальное число эпох |
| `patience` | int | Число эпох без улучшения до остановки |
| `min_delta` | float | Минимальное изменение для учета как улучшения |
| `monitor` | str | Имя метрики для мониторинга ('total_loss', 'pde', и т.д.) |

**Возвращает:**

- `history` — история обучения
- `training_time` — время обучения в секундах

---

## Механизм Early Stopping

Функция `run_experiment` использует Early Stopping для предотвращения переобучения:

1. **Мониторинг метрики**: Отслеживается метрика `monitor` (по умолчанию `total_loss`)
2. **Patience**: Обучение останавливается, если метрика не улучшается в течение `patience` эпох
3. **Min Delta**: Улучшение учитывается только если изменение больше `min_delta`
4. **Восстановление весов**: После остановки восстанавливаются веса лучшей модели

**Пример настройки Early Stopping:**

```python
config = TrainConfig(
    max_epochs=10000,  # Максимум 10000 эпох
    patience=500,  # Остановка после 500 эпох без улучшения
    min_delta=1e-6,  # Минимальное улучшение 1e-6
    monitor="total_loss",  # Мониторим общую потерю
)
```

---

## Формат результатов Grid Search

Результаты сохраняются в CSV файл со следующими колонками:

| Колонка | Описание |
|---------|----------|
| `bc_left` | Значение левого граничного условия |
| `bc_right` | Значение правого граничного условия |
| `training_time` | Время обучения (сек) |
| `epochs_trained` | Количество обученных эпох |
| `lr` | Скорость обучения |
| `activation_func` | Функция активации |
| `layers` | Количество слоёв |
| `neurons` | Количество нейронов |
| `optimizer` | Оптимизатор |
| `collocation_points` | Количество точек коллокации |
| `mape` | MAPE ошибка |
| `mae` | MAE ошибка |
| `mse` | MSE ошибка |
| `rmse` | RMSE ошибка |
| `max_error` | Максимальная ошибка |
| `weights` | Веса потерь |

---

## Полный пример использования

```python
from mpinn.runner import run_experiment, run_grid_search
from mpinn.config import TrainConfig, PhysicsParams
from mpinn.pde import line_1d
from mpinn.analytic import line_1d_robin_exact
import pandas as pd

# === Пример 1: Одиночный эксперимент ===

phys = PhysicsParams(x0=0.0, x1=1.0, T0=300.0, T_inf=500.0, _lambda=1.0, h=10.0)

config = TrainConfig(
    hidden_features=64,
    num_layers=2,
    activation_name="GELU",
    lr=0.01,
    max_epochs=3000,
    patience=200,
    num_points=100,
    weights=(1.0, 1.0, 1.0),
    save_img=True,
    show_plot=False,
    image_path="results/single_experiment.png",
)

metrics, history = run_experiment(
    config=config, phys=phys, pde_fn=line_1d, exact_fn=line_1d_robin_exact
)

print(f"\n=== Результаты эксперимента ===")
print(f"MSE: {metrics['mse']:.6e}")
print(f"MAE: {metrics['mae']:.6e}")
print(f"RMSE: {metrics['rmse']:.6e}")
print(f"Эпох: {metrics['epochs_trained']}")
print(f"Время: {metrics['training_time']}c")


# === Пример 2: Grid Search ===

param_grid = {
    "hidden_features": [32, 64],
    "num_layers": [2, 3],
    "activation_name": ["tanh", "GELU"],
    "lr": [0.001, 0.01],
}

base_config = TrainConfig(max_epochs=2000, patience=150, num_points=80)

df_results = run_grid_search(
    param_grid=param_grid,
    base_config=base_config,
    phys=phys,
    pde_fn=line_1d,
    exact_fn=line_1d_robin_exact,
    csv_path="results/grid_search_results.csv",
)

# Анализ лучших результатов
if not df_results.empty:
    best_idx = df_results["mse"].idxmin()
    best_row = df_results.loc[best_idx]

    print(f"\n=== Лучшая конфигурация ===")
    print(f"Слои: {best_row['layers']}, Нейроны: {best_row['neurons']}")
    print(f"Активация: {best_row['activation_func']}")
    print(f"LR: {best_row['lr']}")
    print(f"MSE: {best_row['mse']:.6e}")
```

---

## Советы по использованию

### Выбор диапазона параметров для Grid Search

```python
# Рекомендуемые диапазоны
param_grid = {
    "hidden_features": [32, 64, 128, 256],  # Размер скрытого слоя
    "num_layers": [2, 3, 4, 5],  # Глубина сети
    "activation_name": ["tanh", "GELU", "Swish"],  # Активация
    "lr": [0.0001, 0.001, 0.01, 0.1],  # Скорость обучения
    "num_points": [50, 100, 200],  # Точки коллокации
}
```

### Настройка Early Stopping

```python
# Для быстрой отладки
config_debug = TrainConfig(max_epochs=500, patience=50, min_delta=1e-4)

# Для финального обучения
config_final = TrainConfig(max_epochs=10000, patience=500, min_delta=1e-7)
```

### Сохранение и визуализация

```python
config = TrainConfig(
    save_img=True,
    show_plot=False,  # False для Grid Search
    image_path="results/best_solution.png",
)
```

---

## См. также

- [config.md](config.md) — Конфигурация
- [pinn_core.md](pinn_core.md) — Ядро PINN
- [plotting.md](plotting.md) — Визуализация
