# Модуль 5 — Современные модели нейронных сетей

> Цель модуля: познакомиться с ключевыми современными архитектурами, которые используются в промышленности и исследованиях: ResNet, Transformer, BERT, GPT, Vision Transformer (ViT). Понять их особенности, отличия и области применения.

---

## 1. ResNet (Residual Networks)

**Идея:** использовать **skip-connections** (остаточные связи), чтобы избежать деградации при обучении очень глубоких сетей.

Формула блока:

```
output = F(x, W) + x
```

Где `F(x, W)` — преобразование (свёртка, активация), `x` — shortcut-путь.

Преимущества:

* Можно обучать сети глубиной 50–1000 слоёв.
* Легче бороться с исчезающими градиентами.

Применения:

* Классификация изображений.
* Обнаружение объектов.
* Сегментация.

Пример кода блока:

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.conv(x) + x)
```

---

## 2. Transformer

**Идея:** заменить рекурренцию механизмом **attention**.

Компоненты:

* **Self-Attention:** каждый токен «смотрит» на все остальные.
* **Multi-Head Attention:** несколько параллельных «взглядов».
* **Feedforward слой:** MLP после внимания.
* **Residual + LayerNorm:** для стабилизации.

Формула внимания:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Преимущества:

* Параллелизация обучения.
* Длинный контекст.
* Гибкость (NLP, CV, мультимодальность).

---

## 3. BERT (Bidirectional Encoder Representations from Transformers)

**Идея:** предобученный энкодер на маскированном языковом моделировании.

Особенности:

* Маскируем часть слов, сеть предсказывает их.
* Двунаправленный контекст (смотрим налево и направо).

Применения:

* Классификация текста.
* Named Entity Recognition (NER).
* Поиск и QA-системы.

Пример использования (HuggingFace):

```python
from transformers import BertTokenizer, BertModel

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

---

## 4. GPT (Generative Pretrained Transformer)

**Идея:** автогрегрессивная модель → предсказывает следующее слово.

Особенности:

* Использует только **decoder** часть трансформера.
* Обучается на задаче language modeling: `P(w_t | w_1, ..., w_{t-1})`.

Применения:

* Генерация текста.
* Чат-боты.
* Кодогенерация.

Ключевая разница с BERT:

* BERT = энкодер, предсказывает маски.
* GPT = декодер, предсказывает следующее слово.

---

## 5. Vision Transformer (ViT)

**Идея:** применить трансформеры к картинкам.

Механизм:

* Изображение режется на **патчи** (например, 16×16).
* Каждый патч линеаризуется и подаётся как «токен».
* Добавляется позиционная информация.
* Дальше обычный Transformer Encoder.

Преимущества:

* Работает лучше CNN при большом количестве данных.
* Универсальность.

Пример (HuggingFace):

```python
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import requests

model = ViTModel.from_pretrained("google/vit-base-patch16-224")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

---

## 6. Сравнительная таблица

| Архитектура | Идея                                  | Применения                       |
| ----------- | ------------------------------------- | -------------------------------- |
| ResNet      | Skip-connections, очень глубокие CNN  | CV: классификация, сегментация   |
| Transformer | Attention вместо рекурренции          | NLP, CV, мультимодальность       |
| BERT        | Энкодер-трансформер, MLM-предобучение | NLP: классификация, QA           |
| GPT         | Декодер-трансформер, авторегрессия    | NLP: генерация, чат-боты         |
| ViT         | Патчи + Transformer                   | CV: классификация, распознавание |

---

## 7. Практика

* Возьмите предобученный ResNet (torchvision.models.resnet18) и дообучите на CIFAR-10.
* Примените BERT для классификации тональности.
* Примените GPT-2 для генерации текста.
* Попробуйте ViT на изображениях.

---

## 8. Чек-лист по итогам модуля

* [ ] Я понимаю идею skip-connections в ResNet.
* [ ] Знаю разницу между Transformer, BERT и GPT.
* [ ] Понимаю, как ViT обрабатывает картинки.
* [ ] Могу применить предобученные модели через HuggingFace.

---

## 9. Домашнее задание

1. Дообучите ResNet-18 на CIFAR-10, сравните точность с MLP.
2. Сравните BERT и GPT: запустите оба и объясните разницу в выходе.
3. Попробуйте ViT и CNN на одной задаче (например, классификация собак/кошек).

---

## 10. Мини-FAQ

**Почему ResNet так важен?**
Он сделал возможным обучение очень глубоких сетей.

**В чём магия трансформеров?**
В attention: модель видит все связи между элементами.

**Почему ViT работает только при больших данных?**
Потому что у трансформеров нет inductive bias свёрток, им нужно много данных для обучения.

---

## 11. Что дальше

* Модуль 6: практика классификации сетей, шпаргалка «задача → архитектура».
* Модуль 7: итоговый проект (сборка модели под задачу).
