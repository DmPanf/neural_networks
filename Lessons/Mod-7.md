# Модуль 7 — Итоговый проект по нейронным сетям

> Цель модуля: применить полученные знания на практике — пройти полный цикл работы с нейросетью: от постановки задачи до обучения, оценки и улучшения модели.

---

## 1. Постановка задачи

1. Определяем **тип задачи**: классификация, регрессия, генерация, сегментация и т.д.
2. Определяем **тип данных**: изображения, текст, звук, табличные данные.
3. Формулируем **метрики качества**: accuracy, F1, MAE, BLEU и др.
4. Ограничения: вычислительные ресурсы, время обучения, доступность данных.

---

## 2. Подготовка данных

* **Сбор:** открытые датасеты (MNIST, CIFAR-10, IMDB, COCO, LibriSpeech и др.) или собственные.
* **Очистка:** удаление дубликатов, исправление ошибок.
* **Аугментация:** повороты, шумы, синонимы, маскирование.
* **Разделение:** train / validation / test.

Пример для изображений:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
```

---

## 3. Выбор архитектуры

Ориентируемся на шпаргалку из модуля 6:

* Изображения → CNN или ViT.
* Текст → BERT (анализ) или GPT (генерация).
* Временные ряды → LSTM, GRU.
* Мультимодальные → CLIP, LLaVA.

---

## 4. Построение модели

### 🔹 Вариант A — классификация изображений (CNN)

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 10)
        )
    def forward(self, x):
        return self.net(x)
```

### 🔹 Вариант B — классификация текста (BERT)

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

---

## 5. Обучение

1. Определяем функцию потерь (CrossEntropy, MSE, BCE).
2. Выбираем оптимизатор (Adam, SGD).
3. Запускаем обучение с батчами.
4. Отслеживаем метрики на train/val.

Пример цикла:

```python
for epoch in range(epochs):
    model.train()
    for X, y in trainloader:
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
```

---

## 6. Оценка модели

* Accuracy, Precision, Recall, F1.
* Матрица ошибок.
* Визуализация (например, Grad-CAM для CNN).

---

## 7. Улучшение модели

* **Регуляризация:** Dropout, weight decay.
* **BatchNorm:** стабилизация обучения.
* **Data augmentation.**
* **Transfer learning:** использование предобученных моделей.
* **Hyperparameter tuning:** lr, batch_size, количество слоёв.

---

## 8. Практика — итоговый пайплайн (пример: классификация CIFAR-10)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Данные
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Модель
model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
for epoch in range(5):
    for X, y in trainloader:
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
```

---

## 9. Чек-лист по итогам модуля

* [ ] Я умею формулировать задачу и метрики.
* [ ] Знаю, как подготовить данные (очистка, аугментация, split).
* [ ] Умею выбрать архитектуру под задачу.
* [ ] Могу построить и обучить модель.
* [ ] Умею оценивать качество и улучшать сеть.

---

## 10. Домашнее задание

1. Выберите задачу (например, классификация картинок или анализ текста).
2. Постройте и обучите модель.
3. Сравните два подхода (например, CNN и ViT, или BERT и LSTM).
4. Сделайте отчёт с графиками и метриками.

---

## 11. Мини-FAQ

**Что делать, если данных мало?**
Использовать transfer learning (дообучение ResNet, BERT).

**Если модель переобучается?**
Dropout, data augmentation, ранняя остановка.

**Если модель недообучается?**
Увеличить слои/нейроны, lr, больше эпох.

---

## 12. Итог

