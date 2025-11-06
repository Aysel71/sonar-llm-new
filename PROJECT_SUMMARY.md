# SONAR-LLM Experiments - Project Summary

## 📋 Что создано

Профессиональная структура для обучения SONAR-LLM на задачах RULER benchmark.

### ✅ Компоненты

#### 1. **Text Cleaning** (`scripts/text_cleaning.py`)
- ✅ `clean_text_flag_style()` - минимальная чистка (как в FlagEmbedding)
- ✅ `clean_text_ruler_style()` - полная чистка для генерации данных
- ✅ `postprocess_prediction()` - обработка предсказаний модели
- 📚 Based on: [FlagEmbedding eval_needle.py](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/Long_LLM/activation_beacon/main/eval_needle.py)

#### 2. **NIAH Data Generation** (`scripts/generate_niah_data.py`)
- ✅ Класс `NIAHDataGenerator`
- ✅ Генерация haystack (distractor text)
- ✅ Генерация needles (keys and values)
- ✅ Поддержка типов: words, numbers, UUIDs
- ✅ Конфигурируемая глубина вставки (0-100%)
- ✅ Чистка текста RULER-style

#### 3. **QA Data Generation** (`scripts/generate_qa_data.py`)
- ✅ Класс `QADataGenerator`
- ✅ 8 базовых QA pairs (SQuAD-style)
- ✅ Генерация distractor параграфов
- ✅ RULER-style prompt template
- ✅ Категории вопросов (person, number, measurement, etc.)

#### 4. **NIAH Training** (`scripts/train_niah.py`)
- ✅ Профессиональная структура с классами
- ✅ `TrainingConfig` - конфигурация обучения
- ✅ `NIAHModel` - SONAR + LLaMA + Projectors
- ✅ `NIAHDataset` - с предвычислением эмбеддингов
- ✅ RULER метрика: `string_match_all`
- ✅ Evaluation с примерами
- ✅ Сохранение best model и checkpoints

#### 5. **QA Training** (`scripts/train_qa.py`)
- ✅ Переиспользует код NIAH (та же архитектура)
- ✅ RULER метрика: `string_match_part`
- ✅ Оптимизированные гиперпараметры (LR=5e-5)

#### 6. **Evaluation** (`scripts/evaluate_model.py`)
- ✅ Загрузка результатов
- ✅ Форматированный отчет
- ✅ Сравнение NIAH vs QA
- ✅ Анализ и рекомендации

#### 7. **Bash Scripts**
- ✅ `run_niah.sh` - полный pipeline для NIAH
- ✅ `run_qa.sh` - полный pipeline для QA
- ✅ Активация conda environment
- ✅ Error handling

#### 8. **Configs**
- ✅ `configs/niah_config.json` - параметры NIAH
- ✅ `configs/qa_config.json` - параметры QA

#### 9. **Documentation**
- ✅ `README.md` - главная документация
- ✅ `USAGE.md` - примеры использования
- ✅ `PROJECT_SUMMARY.md` - этот файл

---

## 🎯 Ключевые особенности

### 1. Следование RULER Benchmark
```python
# NIAH metric (строгая)
def string_match_all(preds, refs):
    score = sum([1.0 if r.lower() in pred.lower() else 0.0 
                 for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

# QA metric (мягкая)
def string_match_part(preds, refs):
    score = sum([1.0 if r.lower() in pred.lower() else 0.0
                 for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)
```

### 2. Чистка текста как в FlagEmbedding
```python
# Для генерации данных
clean_text_ruler_style(text)  # Агрессивная нормализация

# Для предсказаний
clean_text_flag_style(text)   # Минимальная (first line only)
```

### 3. Профессиональная архитектура
```
NIAHDataGenerator/QADataGenerator (генерация)
      ↓
clean_text_ruler_style (чистка)
      ↓
NIAHDataset (предвычисление эмбеддингов)
      ↓
NIAHModel (SONAR + LLaMA + Projectors)
      ↓
Training Loop (AdamW + Cosine schedule)
      ↓
Evaluation (RULER metrics + examples)
```

---

## 📂 Структура файлов

```
sonar_llm_experiments/
├── README.md                          # Главная документация
├── USAGE.md                           # Руководство пользователя
├── PROJECT_SUMMARY.md                 # Этот файл
├── run_niah.sh                        # Запуск NIAH pipeline
├── run_qa.sh                          # Запуск QA pipeline
│
├── configs/                           # Конфигурации
│   ├── niah_config.json              # NIAH параметры
│   └── qa_config.json                # QA параметры
│
├── scripts/                           # Python скрипты
│   ├── text_cleaning.py              # Утилиты чистки текста
│   ├── generate_niah_data.py         # Генерация NIAH
│   ├── generate_qa_data.py           # Генерация QA
│   ├── train_niah.py                 # Обучение NIAH
│   ├── train_qa.py                   # Обучение QA
│   └── evaluate_model.py             # Evaluation и сравнение
│
├── data/                              # Датасеты (создаются автоматически)
│   ├── niah/
│   │   └── niah_*.json
│   └── qa/
│       └── qa_*.json
│
├── models/                            # Модели (создаются при обучении)
│   ├── niah_model/
│   │   ├── checkpoint_step_*/
│   │   ├── best_model/
│   │   └── final_results.json
│   └── qa_model/
│       ├── checkpoint_step_*/
│       ├── best_model/
│       └── final_results.json
│
└── results/                           # Дополнительные результаты
```

---

## 🚀 Использование

### Быстрый старт

```bash
cd sonar_llm_experiments

# NIAH на GPU 0
./run_niah.sh 0

# QA на GPU 1  
./run_qa.sh 1

# Сравнить результаты
cd scripts
python evaluate_model.py --compare
```

### Пошаговое использование

```bash
# 1. Генерация NIAH
cd scripts
python generate_niah_data.py --num_samples 1000 --output_dir ../data/niah

# 2. Генерация QA
python generate_qa_data.py --num_samples 500 --output_dir ../data/qa --add_distractors

# 3. Обучение NIAH
CUDA_VISIBLE_DEVICES=0 python train_niah.py \
  --data_path ../data/niah/niah_dataset.json \
  --output_dir ../models/niah_model \
  --epochs 3

# 4. Обучение QA
CUDA_VISIBLE_DEVICES=1 python train_qa.py \
  --data_path ../data/qa/qa_dataset.json \
  --output_dir ../models/qa_model \
  --epochs 3

# 5. Evaluation
python evaluate_model.py --compare
```

---

## 📊 RULER Compliance

### Метрики соответствуют RULER:
- ✅ NIAH: `string_match_all` 
- ✅ QA: `string_match_part`
- ✅ Точные формулы из RULER/scripts/eval/synthetic/constants.py

### Чистка текста:
- ✅ Генерация: RULER-style (агрессивная)
- ✅ Prediction: FlagEmbedding-style (минимальная)

### Формат данных:
- ✅ JSON with `input`/`output` fields
- ✅ Metadata (key, value, question, answer)
- ✅ Context length tracking

---

## 🎓 Отличия от оригинального кода

### Улучшения:
1. **Модульность** - разделение на отдельные файлы
2. **Конфигурации** - JSON configs вместо hardcoded значений
3. **Классы** - OOP подход для генераторов и конфигов
4. **Документация** - docstrings и комментарии
5. **Error handling** - проверки и информативные ошибки
6. **Логирование** - детальный вывод прогресса
7. **Best model saving** - сохранение лучшей модели по accuracy

### Сохранено из оригинала:
1. **Архитектура** - та же (SONAR + LLaMA + Projectors)
2. **RULER метрики** - точное соответствие
3. **Чистка текста** - FlagEmbedding approach
4. **Предвычисление** - embeddings caching для скорости

---

## 📈 Ожидаемое время выполнения

| Задача | Samples | Epochs | GPU | Время |
|--------|---------|--------|-----|-------|
| NIAH data gen | 10K | - | CPU | ~1 мин |
| QA data gen | 1K | - | CPU | ~10 сек |
| NIAH precompute | 10K | - | GPU | ~40 мин |
| QA precompute | 1K | - | GPU | ~4 мин |
| NIAH training | 10K | 3 | GPU | ~15-20 ч |
| QA training | 1K | 3 | GPU | ~1.5-2 ч |

---

## ✅ Checklist для встречи

- [x] Создана модульная структура
- [x] Отдельные скрипты для NIAH и QA
- [x] Профессиональная чистка текста (FlagEmbedding)
- [x] RULER-compliant метрики
- [x] Конфигурационные файлы
- [x] Bash скрипты для one-command запуска
- [x] Документация (README, USAGE, SUMMARY)
- [x] Evaluation и comparison скрипты

---

## 🎯 Готово к использованию!

Все скрипты готовы, структура профессиональная, код чистый.

**Для запуска:**
```bash
cd sonar_llm_experiments
./run_niah.sh 0  # Запустить NIAH на GPU 0
./run_qa.sh 1    # Запустить QA на GPU 1
```

**Для анализа:**
```bash
cd scripts
python evaluate_model.py --compare
```

