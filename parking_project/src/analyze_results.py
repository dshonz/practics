# analyze_results.py - Анализ результатов экспериментов

import json
import pandas as pd
from pathlib import Path

# Загружаем результаты
with open('reports/experiment_results.json', 'r') as f:
    results = json.load(f)

# Создаем DataFrame
df = pd.DataFrame([
    {
        'Модель': name,
        'Accuracy': data['test_metrics']['accuracy'],
        'Precision': data['test_metrics']['precision'],
        'Recall': data['test_metrics']['recall'],
        'F1': data['test_metrics']['f1'],
        'Скорость (мс)': data['avg_inference_time_ms'],
        'Размер (МБ)': data['model_size_mb']
    }
    for name, data in results.items()
])

# Вычисляем рекомендации
best_accuracy = df.loc[df['Accuracy'].idxmax()]
fastest = df.loc[df['Скорость (мс)'].idxmin()]
smallest = df.loc[df['Размер (МБ)'].idxmin()]

print("\nАНАЛИЗ ЭКСПЕРИМЕНТОВ")
print("="*60)

print("\n ЛУЧШАЯ ПО ТОЧНОСТИ:")
print(f"  {best_accuracy['Модель']}: {best_accuracy['Accuracy']:.2f}%")

print("\n САМАЯ БЫСТРАЯ:")
print(f"  {fastest['Модель']}: {fastest['Скорость (мс)']:.2f} мс")

print("\n САМАЯ ЛЕГКАЯ:")
print(f"  {smallest['Модель']}: {smallest['Размер (МБ)']:.2f} МБ")

print("\n💡 РЕКОМЕНДАЦИЯ:")
if best_accuracy['Accuracy'] - fastest['Accuracy'] > 1.5:
    print(f"  Для промышленного использования рекомендуем {best_accuracy['Модель']}")
    print(f"  (высокая точность важнее скорости)")
else:
    print(f"  Рекомендуем {fastest['Модель']} как лучший баланс")
    print(f"  (скорость и точность на высоком уровне)")

print("\n СТАТИСТИКА:")
print(f"  Медианная точность: {df['Accuracy'].median():.2f}%")
print(f"  Лучшая точность: {df['Accuracy'].max():.2f}%")
print(f"  Худшая точность: {df['Accuracy'].min():.2f}%")
print(f"  Разброс: {df['Accuracy'].max() - df['Accuracy'].min():.2f}%")