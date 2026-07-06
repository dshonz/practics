import os
import shutil
import random
from pathlib import Path

print(" Создание датасета из 3000 изображений...")

source_dir = Path("data/processed")
target_dir = Path("data/dataset_3k")

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

print(f"  Train: {len(train_data)}")
print(f"  Val: {len(val_data)}")
print(f"  Test: {len(test_data)}")

# Копируем файлы
for split, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    print(f"  Копируем {split}...")
    for img_path, cls in data:
        dest = target_dir / split / cls / img_path.name
        shutil.copy2(img_path, dest)

print(" Готово!")
print(f"  Train: {len(list((target_dir/'train'/'Empty').glob('*.jpg')))} Empty, {len(list((target_dir/'train'/'Occupied').glob('*.jpg')))} Occupied")
print(f"  Val: {len(list((target_dir/'val'/'Empty').glob('*.jpg')))} Empty, {len(list((target_dir/'val'/'Occupied').glob('*.jpg')))} Occupied")
print(f"  Test: {len(list((target_dir/'test'/'Empty').glob('*.jpg')))} Empty, {len(list((target_dir/'test'/'Occupied').glob('*.jpg')))} Occupied")