# 🅿️ Определение занятости парковочного места

---

### Датасет

- PKLot (Póvoa de Varzim Parking Lot Dataset)

---

## 📁 Структура проекта

```
parking_project/
├── data/
│   ├── raw/                    # Исходный датасет PKLot
│   └── processed/              # Вырезанные патчи 
│       ├── train/              # 80 000 патчей
│       │   ├── Empty/          # Свободные места
│       │   └── Occupied/       # Занятые места
│       ├── val/                # 10 000 патчей
│       │   ├── Empty/
│       │   └── Occupied/
│       └── test/               # 10 000 патчей
│           ├── Empty/
│           └── Occupied/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Загрузка и предобработка данных
│   ├── model_utils.py          # 5 архитектур нейронных сетей
│   ├── compare_models.py       # Обучение и сравнение моделей
│   ├── train.py                # Обучение одной модели
│   └── analyze_results.py      # Анализ результатов экспериментов
├── models/                     # Сохраненные веса моделей (.pth)
│   ├── resnet18_best.pth
│   ├── custom_cnn_best.pth
│   ├── alexnet_best.pth
│   ├── squeezenet_best.pth
│   └── mobilenet_v2_best.pth
├── reports/                    # Результаты экспериментов
│   ├── experiment_results.json # Все метрики моделей
│   ├── comparison_table.csv    # Таблица сравнения в CSV
│   ├── comparison_table.xlsx   # Таблица сравнения в Excel
│   ├── comparison_charts.png   # Графики сравнения
│   ├── history.json            # История ручных проверок
│   └── dataset_stats.json      # Статистика по датасету
├── app/
│   └── app.py                  # Веб-приложение на Streamlit
├── examples/                   # Примеры успешной/ошибочной работы
│   ├── success/
│   └── errors/
├── requirements.txt            # Зависимости проекта

```

---

## 🚀 Установка и запуск

### 1. Требования

- Python 3.9 или выше
- 8+ ГБ свободной памяти (для обработки датасета)


### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Подготовка данных

```bash
# Скачивание датасета PKLot
# Извлечение патчей и подготовка структуры
python3 src/data_loader.py
```

**ИЛИ** используйте готовый скрипт для создания уменьшенного датасета (3000 патчей для быстрого теста):

```bash
python3 src/prepare_3k_dataset.py
```

### 4. Обучение моделей

```bash
# Обучение и сравнение всех 5 архитектур
python3 src/compare_models.py

# ИЛИ обучение только ResNet-18
python3 src/train.py --model resnet18 --epochs 5
```

### 5. Запуск веб-приложения

```bash
streamlit run app/app.py
```

### 6. Открыть в браузере

Перейдите по адресу: **http://localhost:8501**

---

## 🛠️ Использованные библиотеки

```txt
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
pillow>=9.0.0
streamlit>=1.20.0
openpyxl>=3.0.0
tqdm>=4.64.0
```
---

## 🚀 Быстрый запуск (для проверки)

```bash
# Клонирование
git clone https://github.com/dshonz/parking_project.git
cd parking_project

# Установка
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Подготовка данных (уже скачаны и подготовлены)
# Запуск приложения
streamlit run app/app.py

# Открыть в браузере: http://localhost:8501
```