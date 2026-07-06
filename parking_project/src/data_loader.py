# src/data_loader.py - Загрузка данных для обучения

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path

def get_data_loaders(data_dir="data/processed", batch_size=32, img_size=128):
    
    # Трансформации для тренировки
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Трансформации для валидации и теста
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Загружаем датасеты
    train_dataset = datasets.ImageFolder(
        root=Path(data_dir) / "train",
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=Path(data_dir) / "val",
        transform=val_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=Path(data_dir) / "test",
        transform=val_transform
    )
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    # Возвращаем загрузчики и имена классов
    return train_loader, val_loader, test_loader, train_dataset.classes