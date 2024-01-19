# coding: utf-8


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Rozdział 14. Czas na szczegóły — mechanizm działania biblioteki TensorFlow (3/3)

# Zwróć uwagę, że rozszerzenie zawierające nieobowiązkowy znak wodny stanowi niewielki plugin notatnika IPython / Jupyter, który zaprojektowałem w celu powielania kodu źródłowego. Wystarczy pominąć poniższe wiersze kodu:









# ### Stosowanie estymatorów w klasyfikacji zestawu pisma odręcznego MNIST



BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20
steps_per_epoch = np.ceil(60000 / BATCH_SIZE)




def preprocess(item):
    image = item['image']
    label = item['label']
    image = tf.image.convert_image_dtype(
        image, tf.float32)
    image = tf.reshape(image, (-1,))

    return {'image-pixels':image}, label[..., tf.newaxis]

#Stap 1. Definiowanie funkcji wejściowych (po jednej dla zadania uczenia i ewaluzacji)
## Etap 1. Definiowanie funkcji wejściowych w celu uczenia
def train_input_fn():
    datasets = tfds.load(name='mnist')
    mnist_train = datasets['train']

    dataset = mnist_train.map(preprocess)
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.repeat()

## funkcja wejściowa dla ewaluacji
def eval_input_fn():
    datasets = tfds.load(name='mnist')
    mnist_test = datasets['test']
    dataset = mnist_test.map(preprocess).batch(BATCH_SIZE)
    return dataset




## Etap 2. Kolumna cech
image_feature_column = tf.feature_column.numeric_column(
    key='image-pixels', shape=(28*28))




## Etap 3. Utworzenie wystąpienia estymatora
dnn_classifier = tf.estimator.DNNClassifier(
    feature_columns=[image_feature_column],
    hidden_units=[32, 16],
    n_classes=10,
    model_dir='modele/mnist-dnn/')


## Etap 4. Uczenie i ewaluacja
dnn_classifier.train(
    input_fn=train_input_fn,
    steps=NUM_EPOCHS * steps_per_epoch)




eval_result = dnn_classifier.evaluate(
    input_fn=eval_input_fn)

print(eval_result)


# ### Tworzenie niestandardowego estymatora z istniejącego modelu Keras



## Wyznaczamy ziarno losowości w celu odtwarzania wyników
tf.random.set_seed(1)
np.random.seed(1)

## Tworzy dane
x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1]<0] = 0

x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]




## Etap 1. Definiowanie funkcji wejściowych
def train_input_fn(x_train, y_train, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'cechy-wej':x_train}, y_train.reshape(-1, 1)))

    # tasuje, powtarza przykłady i tworzy ich grupy.
    return dataset.shuffle(100).repeat().batch(batch_size)

def eval_input_fn(x_test, y_test=None, batch_size=8):
    if y_test is None:
        dataset = tf.data.Dataset.from_tensor_slices(
            {'cechy-wej':x_test})
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            ({'cechy-wej':x_test}, y_test.reshape(-1, 1)))


    # tasuje, powtarza przykłady i tworzy ich grupy.
    return dataset.batch(batch_size)




## Etap 2. Definiowanie kolumn cech
features = [
    tf.feature_column.numeric_column(
        key='cechy-wej:', shape=(2,))
]
    
features




## Etap 3. Utworzenie estymatora: konwersja z modelu Keras
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,), name='cechy-wej'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

my_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model,
    model_dir='modele/estymator-funkcji-XOR/')




## Etap 4. Stosowanie estymatora: uczenia/ewaluacja/predykcja

num_epochs = 200
batch_size = 2
steps_per_epoch = np.ceil(len(x_train) / batch_size)

my_estimator.train(
    input_fn=lambda: train_input_fn(x_train, y_train, batch_size),
    steps=num_epochs * steps_per_epoch)




my_estimator.evaluate(
    input_fn=lambda: eval_input_fn(x_valid, y_valid, batch_size))


# ...

# # Podsumowanie

# ...

# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.




