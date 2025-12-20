# Final report (draft)

## 1) Данные
Датасет: `data/raw/ObesityDataSet.csv`  
Таргет: `NObeyesdad` (multiclass).

Пропусков нет, поэтому обработка фокусируется на корректном кодировании признаков.

## 2) Единая предобработка (`src/data_preprocessing.py`)
- Дискретные категориальные (в исходнике float): `FCVC, NCP, CH2O, FAF, TUE`  
  → округление и приведение к `int`.
- Бинарные `yes/no` → `0/1`:
  - `family_history_with_overweight, FAVC, SMOKE, SCC`
- Ординальные:
  - `CAEC, CALC`: `no < Sometimes < Frequently < Always` → `0..3`
- Номинальные:
  - `Gender, MTRANS` → OneHot (в пайплайне)

## 3) Проверка гипотез: 6 вариантов датасета (`data/processed/`)
Генерация: `save_variants(df, "data/processed")`.

1. Полный (рост+вес)
2. С BMI (вместо роста/веса)
3. Без роста
4. Без веса
5. Без роста и веса
6. Только рост и вес

## 4) Модели
Сравнение моделей — только через CV (метрика: `f1_macro`).  
Кандидаты: LogisticRegression, RandomForest, SVC(RBF).

Финальная модель выбирается автоматически скриптом `src/train.py`
по лучшему среднему `f1_macro` на 5-fold CV.

## 5) Подбор гиперпараметров (статус)
Сделано:
- GridSearchCV для LogisticRegression (penalty/C/solver) — в `notebooks/model_experiments.ipynb`.

Планируется:
- Hyperopt для SVM / деревьев решений (или fallback на RandomizedSearchCV).
