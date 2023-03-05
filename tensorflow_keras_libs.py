# Библиотеки TensorFlow и Keras
from tensorflow.keras.models import Sequential             # Подключение класса для создания последовательной модели НС прямого распространения
from tensorflow.keras.layers import Dense                  # Линейный (полносвязный) слой отвечает за соединение нейронов из предыдущего и следующего слоя
from tensorflow.keras.layers import Flatten                # Для преобразования карты признаков в плоский вектор (2D-данные -> 1D), чтобы подготовить входные данные для полносвязной НС
from tensorflow.keras.layers import Dropout                # Слой регуляризации, обнуляющий в модели НС часть весов
from tensorflow.keras.layers import SpatialDropout2D       # Слой регуляризации CNN - для двумерной выборки (размерность [samples, channels, height, width]) -потери двумерных признаков
                                                           # .. предназначен для генерации различных вариаций изображений исходной выборки (чтобы снизить переобучение)

from tensorflow.keras.models import Model                  # Импортируем модели keras: Model
from tensorflow.keras.layers import Input                  # Слой Input используется для создания входных данных модели
from tensorflow.keras.layers import Conv2DTranspose        # Слой для обратного преобразования одного изображения в другое (up-sampling) или генерации нового с низким разрешением
from tensorflow.keras.layers import UpSampling2D           # Слой для увеличения размера изображения в прецедентной модели глубокого обучения (неуправляемое масштабирование)
from tensorflow.keras.layers import concatenate            # Для совмещения двух или более входных тензоров для создания одного единственного выходного тензора (более мощной модели)

from tensorflow.keras.layers import Activation             # Слой активации, который применяет некоторую (обычно) несложную функцию к каждой точке входного тензора
from tensorflow.keras.layers import BatchNormalization     # Нормализация для уменьшения изменчивости данных (помочь сети обучаться быстрее, избежать затухания градиента и переобучения)
from tensorflow.keras.layers import Conv2D                 # Сверточный 2D-слой (Conv2D - это ядро свертки, которое используется для применения фильтра к изображениям)
from tensorflow.keras.layers import MaxPooling2D           # Слой масштабирования (пулинга - операция для уменьшения размерности входных данных подвыборки)

from tensorflow.keras.optimizers import Adam               # Подключение оптимизатора [Adam] (адаптивный момент) - алгоритм класса градиентного спуска и метода Ньютона # [Adadelta] 
from tensorflow.keras import utils                         # Подключение утилит для подготовки данных (to_categorical)
from tensorflow.keras.preprocessing.image import load_img  # Метод для загрузки изображений из файла и преобразования в массив пикселей с помощью PIL (Python Imaging Library)
from PIL import Image                                      # PIL (Python Imaging Library) - для обработки и наложения эффектов на изображения, для преобразования в нейросеть
from tensorflow.keras.preprocessing import image           # Библиотеки для предобработки изображений, необходимые для подачи их в нейронную сеть 
                                                           # .. позволяет читать, изменять размеры, масштабировать пиксели и преобразовывать изображения в более удобные представления

from tensorflow.keras.datasets import mnist                # Библиотека с базой Mnist (набор данных рукописных цифр)
from sklearn.datasets import load_wine                     # Библиотека с базой вин
from keras.datasets import fashion_mnist                   # Библиотека с базой Fashion_Mnist (для загрузки датасета)
from sklearn.model_selection import train_test_split       # Библиотека для разделения данных на выборки
from sklearn.model_selection import RandomizedSearchCV     # Библиотека для оптимизации гиперпараметров с помощью случайного поиска
from sklearn import preprocessing                          # Предварительная обработка данных - для нормализации, масштабирования, шкалирования и интерполяции данных
from sklearn.metrics import accuracy_score                 # Правильность прогнозирования - для оценки правильности прогноза модели на основании фактических результатов
