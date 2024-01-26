# coding: utf-8


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Rozdział 14. Czas na szczegóły — mechanizm działania biblioteki TensorFlow (1/3)

# Zwróć uwagę, że rozszerzenie zawierające nieobowiązkowy znak wodny stanowi niewielki plugin notatnika IPython / Jupyter, który zaprojektowałem w celu powielania kodu źródłowego. Wystarczy pominąć poniższe wiersze kodu:









# ## Cechy kluczowe TensorFlow
# 
# ### Grafy obliczeniowe TensorFlow: migracja do wersji TensorFlow 2
# 
# ### Grafy obliczeniowe
# 
# 





# ### Tworzenie grafu w wersji TensorFlow 1.x
# 
# 







## Graf z wersji TensorFlow 1.x

g = tf.Graph()
with g.as_default():
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    c = tf.constant(3, name='c')
    z = 2*(a - b) + c
    
with tf.compat.v1.Session(graph=g) as sess:
    print('Wynik: z =', sess.run(z))
    print('Wynik: z =', z.eval())


# ### Migracja grafu do wersji TensorFlow 2



## Graf z wersji TensorFlow 2
a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
c = tf.constant(3, name='c')

z = 2*(a - b) + c
tf.print('Wynik: z =', z)


# ### Wczytywanie danych wejściowych do modelu: TensorFlow 1.x



## Graf z wersji TensorFlow 1
g = tf.Graph()
with g.as_default():
    a = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_a')
    b = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_b')
    c = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_c')
    z = 2*(a - b) + c
    
with tf.compat.v1.Session(graph=g) as sess:
    feed_dict = {a:1, b:2, c:3}
    print('Wynik: z =', sess.run(z, feed_dict=feed_dict))


# ### Wczytywanie danych wejściowych do modelu: TensorFlow 2



## TF-v2 style
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z

tf.print('Skalarne dane wejściowe:', compute_z(1, 2, 3))
tf.print('Dane wejściowe rzędu 1.:', compute_z([1], [2], [3]))
tf.print('Dane wejściowe rzędu 2.:', compute_z([[1]], [[2]], [[3]]))


# ### Poprawianie wydajności obliczeniowej za pomocą dekoratorów funkcji



@tf.function
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z

tf.print('Skalarne dane wejściowe:', compute_z(1, 2, 3))
tf.print('Dane wejściowe rzędu 1.:', compute_z([1], [2], [3]))
tf.print('Dane wejściowe rzędu 2.:', compute_z([[1]], [[2]], [[3]]))




@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.int32),))
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z

tf.print('Dane wejściowe rzędu 1.:', compute_z([1], [2], [3]))
tf.print('Dane wejściowe rzędu 1.:', compute_z([1, 2], [2, 4], [3, 6]))


# ```python
# ## oczekujemy, że realizacja poniższego wiersza zakończy się wyświetleniem błędu
# tf.print('Dane wejściowe rzędu 2.:', compute_z([[1], [2]], [[2], [4]], [[3], [6]]))
# 
# 
# ## >> Error:
# #ValueError: Python inputs incompatible with input_signature: 
# #inputs (([[1], [2]], [[2], [4]], [[3], [6]])), input_signature 
# #((TensorSpec(shape=(None,), dtype=tf.int32, name=None), 
# #  TensorSpec(shape=(None,), dtype=tf.int32, name=None), 
# #  TensorSpec(shape=(None,), dtype=tf.int32, name=None)))
# ```



tf.TensorSpec(shape=[None], dtype=tf.int32)


# ## Obiekty Variable służące do przechowywania i aktualizowania parametrów modelu



a = tf.Variable(initial_value=3.14, name='zmienna_a')
b = tf.Variable(initial_value=[1, 2, 3], name='zmienna_b')
c = tf.Variable(initial_value=[True, False], dtype=tf.bool)
d = tf.Variable(initial_value=['abc'], dtype=tf.string)
print(a)
print(b)
print(c)
print(d)




a.trainable




w = tf.Variable([1, 2, 3], trainable=False)

print(w.trainable)




print(w.assign([3, 1, 4], read_value=True))
w.assign_add([2, -1, 2], read_value=False)

print(w.value())




tf.random.set_seed(1)
init = tf.keras.initializers.GlorotNormal()

tf.print(init(shape=(3,)))




v = tf.Variable(init(shape=(2, 3)))
tf.print(v)




class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2, 3)), trainable=True)
        self.w2 = tf.Variable(init(shape=(1, 2)), trainable=False)
                
m = MyModule()
print('Wszystkie zmienne modułu: ', [v.shape for v in m.variables])
print('Zmienna modyfikowalna:   ', [v.shape for v in
                                 m.trainable_variables])


# #### Używanie zmiennych w tf.function

# ```python
# 
# ## uruchomienie poniższego wiersza zakończy się komunikatem o błędzie
# ## ==> nie można tworzyć zmiennych wewnątrz funkcji dekorowanej
# 
# @tf.function
# def f(x):
#     w = tf.Variable([1, 2, 3])
# 
# f([1])
# 
# ## ==> skutkuje błędem
# ## ValueError: tf.function-decorated function tried to create variables on non-first call.
# 
# ```




tf.random.set_seed(1)
w = tf.Variable(tf.random.uniform((3, 3)))

@tf.function
def compute_z(x):    
    return tf.matmul(w, x)

x = tf.constant([[1], [2], [3]], dtype=tf.float32)
tf.print(compute_z(x))


# ## Obliczanie gradientów za pomocą różniczkowania automatycznego i klasy GradientTape
# 

# ### Obliczanie gradientów funkcji straty w odniesieniu do zmiennych modyfikowalnych




w = tf.Variable(1.0)
b = tf.Variable(0.5)
print(w.trainable, b.trainable)

x = tf.convert_to_tensor([1.4])
y = tf.convert_to_tensor([2.1])

with tf.GradientTape() as tape:
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

dloss_dw = tape.gradient(loss, w)

tf.print('dS/dw : ', dloss_dw)




# sprawdzanie obliczonego gradientu
#tf.print(-2*x * (-b - w*x + y))

tf.print(2*x * ((w*x + b) - y))


# ### Obliczanie gradientów w odniesieniu do tensorów niemodyfikowalnych
# 
#  Monitorowanie niemodyfikowalnych tensorów za pomocą funkcji `tape.watch()`



with tf.GradientTape() as tape:
    tape.watch(x)
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.square(y - z)

dloss_dx = tape.gradient(loss, x)

tf.print('dS/dx:', dloss_dx)




# sprawdzanie obliczonego gradientu
tf.print(2*w * ((w*x + b) - y))


# ### Przechowywanie zasobów na obliczanie wielu gradientów
# 
# za pomocą atrybutu `persistent=True`



with tf.GradientTape(persistent=True) as tape:
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

dloss_dw = tape.gradient(loss, w)
dloss_db = tape.gradient(loss, b)

tf.print('dS/dw:', dloss_dw)
tf.print('dS/db:', dloss_db)


# #### Aktualizowanie zmiennych: `optimizer.apply_gradients()`



optimizer = tf.keras.optimizers.SGD()

optimizer.apply_gradients(zip([dloss_dw, dloss_db], [w, b]))

tf.print('Zaktualizowana waga:', w)
tf.print('Zaktualizowane obciążenie:', b)


# ## Upraszczanie implementacji popularnych struktur za pomocą interfejsu Keras
# 
# 



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))

## późne tworzenie zmiennych
model.build(input_shape=(None, 4))
model.summary()




## wyświetlanie zmiennych modelu
for v in model.variables:
    print('{:20s}'.format(v.name), v.trainable, v.shape)


# #### Konfigurowanie warstw
# 
#  * Inicjalizatory Keras `tf.keras.initializers`: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/initializers  
#  * Regularyzatory Keras `tf.keras.regularizers`: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/regularizers  
#  * Funkcje aktywacji `tf.keras.activations`: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations  



model = tf.keras.Sequential()

model.add(
    tf.keras.layers.Dense(
        units=16, 
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        bias_initializer=tf.keras.initializers.Constant(2.0)
    ))

model.add(
    tf.keras.layers.Dense(
        units=32, 
        activation=tf.keras.activations.sigmoid,
        kernel_regularizer=tf.keras.regularizers.l1
    ))

model.build(input_shape=(None, 4))
model.summary()


# #### Kompilowanie modelu
# 
#  * Optymalizatory Keras `tf.keras.optimizers`:  https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers
#  * Funkcje straty Keras `tf.keras.losses`: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses
#  * Wskaźniki wydajności Keras `tf.keras.metrics`: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/metrics



model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.Accuracy(), 
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall(),])


# ## Rozwiązywanie problemu klasyfikacji XOR



tf.random.set_seed(1)
np.random.seed(1)

x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1]<0] = 0

x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

fig = plt.figure(figsize=(6, 6))
plt.plot(x[y==0, 0], 
         x[y==0, 1], 'o', alpha=0.75, markersize=10)
plt.plot(x[y==1, 0], 
         x[y==1, 1], '<', alpha=0.75, markersize=10)
plt.xlabel(r'$x_1$', size=15)
plt.ylabel(r'$x_2$', size=15)
plt.show()




model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, 
                                input_shape=(2,), 
                                activation='sigmoid'))

model.summary()




model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])




hist = model.fit(x_train, y_train, 
                 validation_data=(x_valid, y_valid), 
                 epochs=200, batch_size=2, verbose=0)





history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history['loss'], lw=4)
plt.plot(history['val_loss'], lw=4)
plt.legend(['F. straty uczenia', 'F. straty walidacji'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history['binary_accuracy'], lw=4)
plt.plot(history['val_binary_accuracy'], lw=4)
plt.legend(['Dokładność (uczenie)', 'Dokładność (walidacja)'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()




tf.random.set_seed(1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=4, input_shape=(2,), activation='relu'))
model.add(tf.keras.layers.Dense(units=4, activation='relu'))
model.add(tf.keras.layers.Dense(units=4, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.summary()

## kompilacja:
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

## uczenie:
hist = model.fit(x_train, y_train, 
                 validation_data=(x_valid, y_valid), 
                 epochs=200, batch_size=2, verbose=0)

history = hist.history




fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history['loss'], lw=4)
plt.plot(history['val_loss'], lw=4)
plt.legend(['F. straty uczenia', 'F. straty walidacji'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history['binary_accuracy'], lw=4)
plt.plot(history['val_binary_accuracy'], lw=4)
plt.legend(['Dokładność (uczenie)', 'Dokładność (walidacja)'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()


# ## Zwiększenie możliwości budowania modeli za pomocą interfejsu funkcyjnego Keras
# 
# 



tf.random.set_seed(1)

## warstwa wejściowa:
inputs = tf.keras.Input(shape=(2,))

## warstwy ukryte
h1 = tf.keras.layers.Dense(units=4, activation='relu')(inputs)
h2 = tf.keras.layers.Dense(units=4, activation='relu')(h1)
h3 = tf.keras.layers.Dense(units=4, activation='relu')(h2)

## wyjście:
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(h3)

## konstruuje model:
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()




## kompilacja:
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

## uczenie:
hist = model.fit(x_train, y_train, 
                 validation_data=(x_valid, y_valid), 
                 epochs=200, batch_size=2, verbose=0)

## tworzenie wykresu
history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history['loss'], lw=4)
plt.plot(history['val_loss'], lw=4)
plt.legend(['F. straty uczenia', 'F. straty walidacji'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history['binary_accuracy'], lw=4)
plt.plot(history['val_binary_accuracy'], lw=4)
plt.legend(['Dokładność (uczenie)', 'Dokładność (walidacja)'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()


# ## Implementowanie modeli bazujących na klasie Model
# 
# #### Tworzenie podklas: `tf.keras.Model`
# 
#  * Zdefiniuj metodę `__init__()`
#  * Zdefiniuj metodę `call()`



class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_3 = tf.keras.layers.Dense(units=4, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')
        
    def call(self, inputs):
        h = self.hidden_1(inputs)
        h = self.hidden_2(h)
        h = self.hidden_3(h)
        return self.output_layer(h)
    
tf.random.set_seed(1)

## testowanie:
model = MyModel()
model.build(input_shape=(None, 2))

model.summary()

## kompilowanie:
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

## uczenie:
hist = model.fit(x_train, y_train, 
                 validation_data=(x_valid, y_valid), 
                 epochs=200, batch_size=2, verbose=0)

## tworzenie wykresu
history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history['loss'], lw=4)
plt.plot(history['val_loss'], lw=4)
plt.legend(['F. straty uczenia', 'F. straty walidacji'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history['binary_accuracy'], lw=4)
plt.plot(history['val_binary_accuracy'], lw=4)
plt.legend(['Dokładność (uczenie)', 'Dokładność (walidacja)'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()


# ## Pisanie niestandardowych warstw Keras
# 
# 
# #### Definiowanie niestandardowej warstwy:
#  * Zdefiniuj metodę `__init__()`
#  * Zdefiniuj metodę `build()` w celu późnego tworzenia zmiennych
#  * Zdefiniuj metodę `call()`
#  * Zdefiniuj metodę `get_config()` w celu serializacji



class NoisyLinear(tf.keras.layers.Layer):
    def __init__(self, output_dim, noise_stddev=0.1, **kwargs):
        self.output_dim = output_dim
        self.noise_stddev = noise_stddev
        super(NoisyLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='wagi',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='random_normal',
                                 trainable=True)
        
        self.b = self.add_weight(shape=(self.output_dim,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, training=False):
        if training:
            batch = tf.shape(inputs)[0]
            dim = tf.shape(inputs)[1]
            noise = tf.random.normal(shape=(batch, dim),
                                     mean=0.0,
                                     stddev=self.noise_stddev)

            noisy_inputs = tf.add(inputs, noise)
        else:
            noisy_inputs = inputs
        z = tf.matmul(noisy_inputs, self.w) + self.b
        return tf.keras.activations.relu(z)
    
    def get_config(self):
        config = super(NoisyLinear, self).get_config()
        config.update({'output_dim': self.output_dim,
                       'noise_stddev': self.noise_stddev})
        return config


## testowanie:

tf.random.set_seed(1)

noisy_layer = NoisyLinear(4)
noisy_layer.build(input_shape=(None, 4))

x = tf.zeros(shape=(1, 4))
tf.print(noisy_layer(x, training=True))

## odtwarzanie z obiektu config:
config = noisy_layer.get_config()
new_layer = NoisyLinear.from_config(config)
tf.print(new_layer(x, training=True))




tf.random.set_seed(1)

model = tf.keras.Sequential([
    NoisyLinear(4, noise_stddev=0.1),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')])

model.build(input_shape=(None, 2))
model.summary()

## kompilacja:
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

## uczenie:
hist = model.fit(x_train, y_train, 
                 validation_data=(x_valid, y_valid), 
                 epochs=200, batch_size=2, 
                 verbose=0)

## tworzenie wykresu
history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history['loss'], lw=4)
plt.plot(history['val_loss'], lw=4)
plt.legend(['F. straty uczenia', 'F. straty walidacji'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history['binary_accuracy'], lw=4)
plt.plot(history['val_binary_accuracy'], lw=4)
plt.legend(['Dokładność (uczenie)', 'Dokładność (walidacja)'], fontsize=15)
ax.set_xlabel('Epoki', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()


# ...

# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.




