# compare_models.py - Сравнение 5 архитектур нейронных сетей

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
from pathlib import Path
import json
import sys
from tqdm import tqdm

# Подключаем наши модули
sys.path.append(str(Path(__file__).parent / "src"))
from data_loader import get_data_loaders
from model_utils import get_model

# Список моделей для сравнения (5 архитектур)
MODELS_TO_COMPARE = [
    'custom_cnn',      # 1. Своя простая CNN (Baseline)
    'alexnet',         # 2. Классическая AlexNet
    'resnet18',        # 3. ResNet-18 (остаточные связи)
    'squeezenet',      # 4. Легкий SqueezeNet
    'mobilenet_v2'     # 5. MobileNetV2 (для мобильных устройств)
]
def prepare_3k_dataset():
    """Создает датасет из 3000 изображений, если его нет"""
    
    source_dir = Path("data/processed")
    target_dir = Path("data/dataset_3k")
    
    # Если датасет уже существует, пропускаем
    if target_dir.exists() and len(list(target_dir.glob("**/*.jpg"))) >= 3000:
        print(f" Датасет 3K уже существует: {target_dir}")
        return target_dir
    
    print("🔄 Создание датасета из 3000 изображений...")
    
    # Создаем папки
    for split in ['train', 'val', 'test']:
        for cls in ['Empty', 'Occupied']:
            (target_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Собираем все изображения
    all_images = []
    for cls in ['Empty', 'Occupied']:
        for split in ['train', 'val', 'test']:
            src_path = source_dir / split / cls
            if src_path.exists():
                for img in src_path.glob("*.jpg"):
                    all_images.append((img, cls))
    
    if not all_images:
        print("❌ Нет данных! Сначала запустите подготовку PKLot.")
        return None
    
    print(f" Всего найдено: {len(all_images)}")
    
    # Перемешиваем
    random.seed(42)
    random.shuffle(all_images)
    
    # Берем только 3000
    selected = all_images[:3000]
    
    # Разбиваем (70% train, 15% val, 15% test)
    total = len(selected)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)
    
    train_data = selected[:train_end]
    val_data = selected[train_end:val_end]
    test_data = selected[val_end:]
    
    # Копируем файлы
    for split, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        print(f"  Копируем {split}...")
        for img_path, cls in data:
            dest = target_dir / split / cls / img_path.name
            shutil.copy2(img_path, dest)
    
    print(" Датасет 3K создан!")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    return target_dir

class ModelTester:
    """Класс для тестирования и сравнения моделей"""
    

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Используем устройство: {self.device}")
        
        # СОЗДАЕМ ДАТАСЕТ 3K
        data_dir = prepare_3k_dataset()
        
        if data_dir is None:
            print("❌ Не удалось создать датасет. Выход.")
            sys.exit(1)
        
        # Загружаем данные
        self.train_loader, self.val_loader, self.test_loader, self.classes = get_data_loaders(
            data_dir=str(data_dir),
            batch_size=16
        )
        print(f" Данные загружены. Классы: {self.classes}")
        
        # Для хранения результатов
        self.results = {}
        
        # Создаем папки для сохранения
        Path("reports").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

    
    def get_model_size(self, model):
        """Вычисляет размер модели в МБ"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return round(size_mb, 2)
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Обучает одну эоху"""
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(train_loader)
    
    def evaluate(self, model, data_loader):
        """Оценивает модель на данных"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return all_preds, all_labels
    
    def calculate_metrics(self, y_true, y_pred):
        """Вычисляет все метрики"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': round(accuracy_score(y_true, y_pred) * 100, 2),
            'precision': round(precision_score(y_true, y_pred, average='binary') * 100, 2),
            'recall': round(recall_score(y_true, y_pred, average='binary') * 100, 2),
            'f1': round(f1_score(y_true, y_pred, average='binary') * 100, 2)
        }
    
    def test_model(self, model_name, num_epochs=5, batch_size=32):
        """Тестирует одну модель"""
        
        print(f"\n{'='*60}")
        print(f" Тестирование модели: {model_name}")
        print('='*60)
        
        # Создаем модель
        model = get_model(model_name).to(self.device)
        
        # Замеряем размер
        model_size = self.get_model_size(model)
        print(f" Размер модели: {model_size} MB")
        
        # Настройки обучения
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Метрики для обучения
        train_losses = []
        val_accuracies = []
        epoch_times = []
        
        # Обучение
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Обучаем эпоху
            train_loss = self.train_epoch(model, self.train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            
            # Валидация
            val_preds, val_labels = self.evaluate(model, self.val_loader)
            val_metrics = self.calculate_metrics(val_labels, val_preds)
            val_accuracies.append(val_metrics['accuracy'])
            
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Val Acc={val_metrics['accuracy']:.2f}%, Time={epoch_time:.2f}s")
        
        # Тестирование на тестовой выборке
        test_preds, test_labels = self.evaluate(model, self.test_loader)
        test_metrics = self.calculate_metrics(test_labels, test_preds)
        
        # Матрица ошибок
        cm = confusion_matrix(test_labels, test_preds)
        
        # Замеряем время инференса (усредненно)
        inference_times = []
        model.eval()
        with torch.no_grad():
            for _ in range(100):
                # Берем один батч
                images, _ = next(iter(self.test_loader))
                images = images.to(self.device)
                start_time = time.time()
                _ = model(images)
                inference_times.append(time.time() - start_time)
        
        avg_inference_time = np.mean(inference_times) * 1000 / batch_size  # в миллисекундах на одно изображение
        
        # Сохраняем модель
        torch.save(model.state_dict(), f"models/{model_name}_best.pth")
        
        # Сохраняем результаты
        self.results[model_name] = {
            'model_name': model_name,
            'model_size_mb': model_size,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'epoch_times': epoch_times,
            'test_metrics': test_metrics,
            'confusion_matrix': cm.tolist(),
            'avg_inference_time_ms': round(avg_inference_time, 2),
            'num_epochs': num_epochs
        }
        
        print(f"\ Тест завершен для {model_name}")
        print(f"   Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"   Inference time: {avg_inference_time:.2f} ms/image")
        
        return model

    def run_all_experiments(self):
        """Запускает эксперименты для всех моделей"""
        
        print("\ НАЧИНАЕМ ЭКСПЕРИМЕНТЫ С МОДЕЛЯМИ")
        print("="*60)
        
        for model_name in MODELS_TO_COMPARE:
            try:
                self.test_model(model_name)
            except Exception as e:
                print(f" Ошибка при тестировании {model_name}: {e}")
                continue
        
        # Сохраняем все результаты
        self.save_results()
        self.create_comparison_table()
        self.plot_comparison_charts()
        
        print(" Все эксперименты завершены!")
    
    def save_results(self):
        """Сохраняет результаты в JSON"""
        results_path = Path("reports/experiment_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f" Результаты сохранены в {results_path}")
    
    def create_comparison_table(self):
        """Создает таблицу сравнения"""
        
        # Собираем данные
        data = []
        for model_name, res in self.results.items():
            data.append({
                'Архитектура': model_name.replace('_', ' ').title(),
                'Accuracy (%)': res['test_metrics']['accuracy'],
                'Precision (%)': res['test_metrics']['precision'],
                'Recall (%)': res['test_metrics']['recall'],
                'F1-score (%)': res['test_metrics']['f1'],
                'Время (мс)': res['avg_inference_time_ms'],
                'Размер (МБ)': res['model_size_mb']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Accuracy (%)', ascending=False)
        
        # Сохраняем в CSV
        csv_path = Path("reports/comparison_table.csv")
        df.to_csv(csv_path, index=False)
        print(f" Таблица сохранена в {csv_path}")
        
        # Сохраняем в Excel
        try:
            excel_path = Path("reports/comparison_table.xlsx")
            df.to_excel(excel_path, index=False)
            print(f" Таблица сохранена в {excel_path}")
        except:
            print(" Excel не сохранен (openpyxl не установлен)")
        
        # Выводим таблицу
        print("\n ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return df
    
    def plot_comparison_charts(self):
        """Строит графики сравнения"""
        
        if not self.results:
            print("Нет результатов для построения графиков")
            return
        
        # Создаем графики
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Сравнение Accuracy
        ax1 = axes[0, 0]
        names = [res['model_name'].replace('_', ' ').title() for res in self.results.values()]
        accuracies = [res['test_metrics']['accuracy'] for res in self.results.values()]
        bars = ax1.bar(names, accuracies, color='skyblue')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Сравнение точности моделей')
        ax1.set_ylim([0, 100])
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 2. Сравнение времени инференса
        ax2 = axes[0, 1]
        times = [res['avg_inference_time_ms'] for res in self.results.values()]
        bars = ax2.bar(names, times, color='lightcoral')
        ax2.set_ylabel('Время (мс)')
        ax2.set_title('Сравнение скорости инференса')
        for bar, t in zip(bars, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{t:.1f} мс', ha='center', va='bottom')
        
        # 3. Сравнение размера модели
        ax3 = axes[1, 0]
        sizes = [res['model_size_mb'] for res in self.results.values()]
        bars = ax3.bar(names, sizes, color='lightgreen')
        ax3.set_ylabel('Размер (МБ)')
        ax3.set_title('Сравнение размера моделей')
        for bar, s in zip(bars, sizes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{s:.1f} MB', ha='center', va='bottom')
        
        # 4. Графики обучения (Accuracy по эпохам)
        ax4 = axes[1, 1]
        for res in self.results.values():
            ax4.plot(res['val_accuracies'], marker='o', label=res['model_name'].replace('_', ' ').title())
        ax4.set_xlabel('Эпоха')
        ax4.set_ylabel('Validation Accuracy (%)')
        ax4.set_title('Динамика обучения моделей')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig('reports/comparison_charts.png', dpi=300, bbox_inches='tight')
        print(" Графики сохранены в reports/comparison_charts.png")
        
        # Показываем график
        plt.show()

    def print_summary(self):
        """Выводит краткую сводку результатов"""
        
        print("\n СВОДКА РЕЗУЛЬТАТОВ")
        print("="*60)
        
        # Находим лучшую модель по accuracy
        best_acc = max(self.results.values(), key=lambda x: x['test_metrics']['accuracy'])
        # Находим самую быструю
        fastest = min(self.results.values(), key=lambda x: x['avg_inference_time_ms'])
        # Находим самую легкую
        lightest = min(self.results.values(), key=lambda x: x['model_size_mb'])
        
        print(f" Лучшая по точности: {best_acc['model_name']} ({best_acc['test_metrics']['accuracy']:.2f}%)")
        print(f"⚡ Самая быстрая: {fastest['model_name']} ({fastest['avg_inference_time_ms']:.2f} мс)")
        print(f" Самая легкая: {lightest['model_name']} ({lightest['model_size_mb']:.2f} МБ)")
        print("\nРекомендация:")
        
        # Логика выбора
        if best_acc['test_metrics']['accuracy'] - 2 > fastest['test_metrics']['accuracy']:
            print(f" Рекомендуем {best_acc['model_name']} для максимальной точности")
        elif fastest['test_metrics']['accuracy'] > 95:
            print(f" Рекомендуем {fastest['model_name']} для баланса скорость/качество")
        else:
            print(f" Рекомендуем {best_acc['model_name']} как лучший компромисс")


if __name__ == "__main__":
    # Создаем тестер
    tester = ModelTester()
    
    # Запускаем все эксперименты
    tester.run_all_experiments()
    
    # Выводим сводку
    tester.print_summary()