# app/app.py - Веб-приложение для анализа парковки

import streamlit as st
import torch
from PIL import Image
from pathlib import Path
import sys
import json
from datetime import datetime
from collections import Counter

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from model_utils import get_model
    from data_loader import get_data_loaders
except ImportError as e:
    st.error(f"❌ Ошибка импорта: {e}")
    st.stop()

st.set_page_config(page_title="🅿️ Парковочный детектор", layout="wide")

st.title("🅿️ Система мониторинга парковки")
st.markdown("---")

# Загружаем модель
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model('resnet18')
    model_path = Path("models/resnet18_best.pth")
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        return None, None

model, device = load_model()

if model is None:
    st.error("❌ Модель не найдена! Обучите модель: python3 src/compare_models.py")
    st.stop()

# ============================================
# РЕЖИМ 1: РУЧНАЯ ЗАГРУЗКА ФОТО
# ============================================
st.header("📸 Ручная проверка места")

uploaded_file = st.file_uploader(
    "Загрузите фото парковочного места",
    type=['jpg', 'jpeg', 'png'],
    help="Выберите изображение с компьютера"
)

if uploaded_file is not None:
    from torchvision import transforms
    
    # Отображаем фото
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image, caption="Загруженное изображение", width=300)
    
    # Предобработка
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Инференс
    with torch.no_grad():
        outputs = torch.softmax(model(input_tensor), dim=1)
        confidence, predicted = torch.max(outputs, 1)
    
    # Результат
    class_names = ['Свободно 🟢', 'Занято 🔴']
    result = class_names[predicted.item()]
    conf = confidence.item() * 100
    
    with col2:
        st.subheader("📋 Результат:")
        if predicted.item() == 0:
            st.success(f"### {result}")
        else:
            st.error(f"### {result}")
        st.metric("🎯 Уверенность", f"{conf:.1f}%")
        st.progress(int(conf))
    
    # Сохраняем историю
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": "Свободно" if predicted.item() == 0 else "Занято",
        "confidence": round(conf, 2)
    }
    
    Path("reports").mkdir(exist_ok=True)
    history_path = Path("reports/history.json")
    
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(history_entry)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    st.success("✅ Результат сохранен в историю!")

# ============================================
# БЛОК СТАТИСТИКИ (по ручным проверкам)
# ============================================
st.markdown("---")
st.header("📊 Статистика ручных проверок")

history_path = Path("reports/history.json")
if history_path.exists():
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    total = len(history)
    free = sum(1 for h in history if h['prediction'] == 'Свободно')
    occupied = total - free
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🔄 Всего проверок", total)
    col2.metric("🟢 Свободно", free)
    col3.metric("🔴 Занято", occupied)
    
    with st.expander("📜 История проверок (последние 10)"):
        if history:
            for i, entry in enumerate(reversed(history[-10:]), 1):
                st.write(f"{i}. {entry['timestamp']} — **{entry['prediction']}** ({entry['confidence']}%)")
        else:
            st.write("Нет записей")
else:
    st.info("Нет сохраненных проверок. Загрузите фото, чтобы начать!")

# ============================================
# РЕЖИМ 2: СТАТИСТИКА ПО ВСЕМУ ДАТАСЕТУ (3000 фото)
# ============================================
st.markdown("---")
st.header("📊 Статистика по всему датасету (3000 изображений)")

if st.button("🔍 Проанализировать весь датасет", type="primary"):
    with st.spinner("Анализ 3000 изображений..."):
        # Загружаем ВЕСЬ датасет (train + val + test)
        from torch.utils.data import ConcatDataset, DataLoader
        
        try:
            train_loader, val_loader, test_loader, classes = get_data_loaders(
                data_dir="data/dataset_3k",
                batch_size=32
            )
        except:
            train_loader, val_loader, test_loader, classes = get_data_loaders(
                data_dir="data/processed",
                batch_size=32
            )
        
        full_dataset = ConcatDataset([
            train_loader.dataset,
            val_loader.dataset,
            test_loader.dataset
        ])
        
        full_loader = DataLoader(
            full_dataset,
            batch_size=32,
            shuffle=False
        )
        
        results = []
        total = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(full_loader):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                for pred in preds:
                    results.append('Свободно' if pred.item() == 0 else 'Занято')
                    total += 1
                
                progress_bar.progress((i + 1) / len(full_loader))
                status_text.text(f"Обработано: {total} изображений")
        
        progress_bar.empty()
        status_text.empty()
    
    # Подсчет статистики
    counter = Counter(results)
    free = counter.get('Свободно', 0)
    occupied = counter.get('Занято', 0)
    
    # Сохраняем в JSON
    stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "free": free,
        "occupied": occupied,
        "free_percent": round(free / total * 100, 2),
        "occupied_percent": round(occupied / total * 100, 2)
    }
    
    Path("reports").mkdir(exist_ok=True)
    with open("reports/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    # Отображение результатов
    st.success("✅ Анализ завершен!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📸 Всего мест", total)
    col2.metric("🟢 Свободно", free, f"{stats['free_percent']}%")
    col3.metric("🔴 Занято", occupied, f"{stats['occupied_percent']}%")
    
    # Прогресс-бар
    st.subheader("📈 Общая загрузка парковки")
    st.progress(occupied / total)
    st.caption(f"Занято {occupied} из {total} мест ({stats['occupied_percent']}%)")
    
    # Показываем результаты в таблице
    st.subheader("📋 Результаты анализа")
    st.table({
        "Показатель": ["Всего мест", "Свободно", "Занято"],
        "Количество": [total, free, occupied],
        "Процент": ["100%", f"{stats['free_percent']}%", f"{stats['occupied_percent']}%"]
    })
    
else:
    st.info("👆 Нажмите кнопку выше, чтобы проанализировать все 3000 изображений в датасете")