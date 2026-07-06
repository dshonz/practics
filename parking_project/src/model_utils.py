# src/model_utils.py - Все архитектуры моделей

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    AlexNet_Weights,
    ResNet18_Weights,
    SqueezeNet1_0_Weights,
    MobileNet_V2_Weights
)

def get_model(model_name, num_classes=2):
    """Возвращает модель по имени"""
    
    if model_name == 'custom_cnn':
        return CustomCNN(num_classes)
    
    elif model_name == 'alexnet':
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model
    
    elif model_name == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, num_classes)
        return model
    
    elif model_name == 'squeezenet':
        model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        return model
    
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(1280, num_classes)
        return model
    
    else:
        raise ValueError(f"Модель {model_name} не поддерживается")

class CustomCNN(nn.Module):
    """ сверточная сеть"""
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x