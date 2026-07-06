# train_one.py - Быстрое обучение одной модели

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from pathlib import Path

# Подключаем наши модули
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import get_data_loaders
from model_utils import get_model

print("🔥 Начинаем обучение модели ResNet-18...")

# Загружаем данные
train_loader, val_loader, test_loader, classes = get_data_loaders()
print(f" Данные загружены! Классы: {classes}")

# Модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model('resnet18').to(device)
print(f" Модель на {device}")

# Настройки
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5  # 5 эпох для быстрого теста

# Обучение
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Валидация
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Val Acc={acc:.2f}%")

# Сохраняем модель
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/resnet18_best.pth")
print(" Модель сохранена в models/resnet18_best.pth")

# Тест
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f" Тестовая точность: {100 * correct / total:.2f}%")
print(" Обучение завершено!")