# Проектный практикум УРФУ

## Предварительная настройка

После клонирования проекта необходимо сделать следующее

Создать или обновить окружение:
```bash
conda env create -f environment.yml     # если окружения ещё нет
# или
conda env update -f environment.yml     # если окружение уже создано
```
Активировать окружение:
```bash
conda activate obesity-risk
```
Чтобы `import utils` работал в ноутбуках, запускайте Jupyter из этой папки (`shared_folder/vova_folder`) или добавьте её в `PYTHONPATH`.

## Установка зависимостей в Conda

Не обязательно, но упрощает работу - с помощью Conda можно установить разом все необходимые для работы с проектом библитеки. Подходит для macOS, Linux и Windows. Библиотеки с указанием версии перечислены в файле окружения: `environment.yml` - лежит в корне проекта.

### Mac (Linux)

Если не установлена Conda, устанавливаем:
```bash
brew install --cask miniconda
```
Активируем и перезагружаем терминал:
```bash
conda init zsh
exec zsh
```

Создаем окружение из `environment.yml` (нужно делать из папки с этим файлом):
```bash
conda env create -f environment.yml
conda activate obesity-risk
```

- Обновить окружение после правок `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

### Windows

Выполните команды в Anaconda Prompt или в PowerShell с активированной инициализацией conda.

`conda activate obesity-risk` работает в Anaconda Prompt и в PowerShell после `conda init powershell`.


## Подготовка данных

Утилиты загрузки и очистки данных находятся в `utils.py` (в этой папке).

Основные функции:

- `load_raw_df()` - загружает сырой CSV `ObesityDataSet.csv` из директории проекта.
- `load_clean_df()` - загружает и очищает (удаляет дубли и прочее), а также добавляет `BMI` и `NObeyesdad_norm` (если есть маппинг в `columns_mapping.yml`).

Пример запуска (можно использовать и внутри ноутбуков):

```python
from utils import load_raw_df, load_clean_df

df = load_raw_df()
df_clean = load_clean_df()
```

Если в эти функции подставить `'short_names'` или `'long_names'` (алиасы для схем из `columns_mapping.yml`), то будет сформирован датафрейм с русскими названиями столбцов:

```python
from utils import load_raw_df, load_clean_df

df = load_raw_df('short_names')
df_clean = load_clean_df('short_names')
```

Меняйте логику очистки в `load_clean_df` в `utils.py`, если нужно и централизованно.

## Маппинг колонок (`columns_mapping.yml`)

Файл `columns_mapping.yml` - единый источник метаданных для признаков и таргетов: человеко-читаемые названия (RU/EN), группы признаков, порядок категорий (для ординальных признаков/таргета), а также маппинг для агрегирования классов (например, `NObeyesdad_norm`).

Структура маппинга:

- `globals.groups_order` - порядок групп (для сортировки/визуализаций).
- `columns.<feature>` - описание признака (например: `description_ru`, `short_ru`, `group_ru`, `order`).
- `targets.<target>` - описание таргета (например: `order`, `categories`, а для `NObeyesdad_norm` ещё и `mapping`).

Полезные функции из `utils.py` для работы с маппингом:

- `load_columns_mapping()` - загрузить весь YAML как dict.
- `clear_columns_mapping_cache()` - сбросить кеш маппинга (удобно, если правите YAML во время сессии ноутбука).
- `cm_columns()`, `cm_targets()` - получить секции `columns`/`targets`.
- `cm_label(name, key="description_ru")` - получить подпись для колонки/таргета (например, RU описание).
- `cm_group(name)` - получить группу (`group_ru`) для признака/таргета.
- `cm_order(name)` - получить порядок категорий (`order`) для ординального поля.
- `cm_groups_order()` - получить глобальный порядок групп.
- `cm_target_mapping(target)` - получить маппинг классов (например, для `NObeyesdad_norm`).
- `cm_target_categories(target)`, `cm_target_order(target)`, `cm_target_category_label(target, category)` - метаданные и подписи по классам таргета.
- `cm_labels_dict(names, key="description_ru")` - словарь `{имя: подпись}` для списка полей.

Мини-пример:

```python
from utils import cm_label, cm_group, cm_order, cm_target_mapping

print(cm_label("Age"))          # "Возраст"
print(cm_group("FCVC"))         # "Признаки пищевых привычек"
print(cm_order("CH2O"))         # ["<1", "1-2", ">2"]
print(cm_target_mapping("NObeyesdad_norm"))  # агрегирование классов
```
