# coding: utf-8


import tensorflow as tf
import numpy as np
import scipy.signal
import imageio
from tensorflow import keras
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import os

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Rozdział 15. Klasyfikowanie obrazów za pomocą głębokich splotowych sieci neuronowych (1/2)

# Zwróć uwagę, że rozszerzenie zawierające nieobowiązkowy znak wodny stanowi niewielki plugin notatnika IPython / Jupyter, który zaprojektowałem w celu powielania kodu źródłowego. Wystarczy pominąć poniższe wiersze kodu:









# ##  Podstawowe elementy splotowej sieci neuronowej
# 
# ### Splotowe sieci neuronowe i hierarchie cech





# 
# 
# ### Splot dyskretny
# 
# ### Splot dyskretny w jednym wymiarze
# 
# 









# ### Uzupełnianie zerami jako sposób kontrolowania rozmiaru wyjściowych map cech
# 
# 





# ### Określanie rozmiaru wyniku splotu




print('Wersja modułu TensorFlow:', tf.__version__)
print('Wersja modułu NumPy: ', np.__version__)




def conv1d(x, w, p=0, s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate(
            [zero_pad, x_padded, zero_pad])
    res = []
    for i in range(0, int((len(x_padded) - len(w_rot)) / s) + 1, s):
        res.append(np.sum(
            x_padded[i:i+w_rot.shape[0]] * w_rot))
    return np.array(res)


## Testowanie:
x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]

print('Implementacja jednowymiarowego splotu:',
      conv1d(x, w, p=2, s=1))

print('Wyniki uzyskane za pomocą NumPy:',
      np.convolve(x, w, mode='same')) 


# ### Splot dyskretny w dwóch wymiarach

















def conv2d(X, W, p=(0, 0), s=(1, 1)):
    W_rot = np.array(W)[::-1,::-1]
    X_orig = np.array(X)
    n1 = X_orig.shape[0] + 2*p[0]
    n2 = X_orig.shape[1] + 2*p[1]
    X_padded = np.zeros(shape=(n1, n2))
    X_padded[p[0]:p[0]+X_orig.shape[0],
    p[1]:p[1]+X_orig.shape[1]] = X_orig

    res = []
    for i in range(0, int((X_padded.shape[0] - 
                           W_rot.shape[0])/s[0])+1, s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1] - 
                               W_rot.shape[1])/s[1])+1, s[1]):
            X_sub = X_padded[i:i+W_rot.shape[0],
                             j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub * W_rot))
    return(np.array(res))

X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]

print('Implementacja splotu dwuwymiarowego: \n',
    conv2d(X, W, p=(1, 1), s=(1, 1)))


print('Wyniki uzyskane w module SciPy:\n',
    scipy.signal.convolve2d(X, W, mode='same'))


# ## Warstwy podpróbkowania





# ## Implementowanie sieci CNN
# 
# ### Praca z wieloma kanałami wejściowymi/barw
# 
# 





# **WSKAZÓWKA: Odczytywanie pliku obrazu**





img_raw = tf.io.read_file('przyk-obraz.png')
img = tf.image.decode_image(img_raw)
print('Rozmiar obrazu:', img.shape)
print('Liczba kanałów:', img.shape[2])
print('Typ danych obrazu:', img.dtype)
print(img[100:102, 100:102, :])






img = imageio.imread('przyk-obraz.png')
print('Rozmiar obrazu:', img.shape)
print('Liczba kanałów:', img.shape[2])
print('Typ danych obrazu:', img.dtype)
print(img[100:102, 100:102, :])


# **RAMKA: rząd obrazu czarno-białego na wejściu sieci splotowej**



img_raw = tf.io.read_file('przyk-obraz-szary.png')
img = tf.image.decode_image(img_raw)
tf.print('Rząd:', tf.rank(img))
tf.print('Wymiary:', img.shape)




img = imageio.imread('przyk-obraz-szary.png')
tf.print('Rząd:', tf.rank(img))
tf.print('Wymiary:', img.shape)

img_reshaped = tf.reshape(img, (img.shape[0], img.shape[1], 1))
tf.print('Nowe wymiary:', img_reshaped.shape)


# ## Regularyzowanie sieci neuronowej metodą porzucania
# 
# 









conv_layer = keras.layers.Conv2D(
    filters=16, kernel_size=(3, 3),
    kernel_regularizer=keras.regularizers.l2(0.001))

fc_layer = keras.layers.Dense(
    units=16, kernel_regularizer=keras.regularizers.l2(0.001))


# ## Funkcje straty w zadaniach klasyfikacji
# 
#  * **`BinaryCrossentropy()`**
#    * `from_logits=False` 
#    * `from_logits=True`
# 
#  * **`CategoricalCrossentropy()`**
#    * `from_logits=False`
#    * `from_logits=True`
#    
#  * **`SparseCategoricalCrossentropy()`**
#    * `from_logits=False`
#    * `from_logits=True`
# 







####### Binarna entropia krzyżowa
bce_probas = tf.keras.losses.BinaryCrossentropy(from_logits=False)
bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

logits = tf.constant([0.8])
probas = tf.keras.activations.sigmoid(logits)

tf.print(
    'BCE (dla Probas): {:.4f}'.format(
    bce_probas(y_true=[1], y_pred=probas)),
    '(dla Logits): {:.4f}'.format(
    bce_logits(y_true=[1], y_pred=logits)))


####### Kategorialna entropia krzyżowa
cce_probas = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False)
cce_logits = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True)

logits = tf.constant([[1.5, 0.8, 2.1]])
probas = tf.keras.activations.softmax(logits)

tf.print(
    'CCE (dla Probas): {:.4f}'.format(
    cce_probas(y_true=[0, 0, 1], y_pred=probas)),
    '(dla Logits): {:.4f}'.format(
    cce_logits(y_true=[0, 0, 1], y_pred=logits)))

####### Rzadka, kategorialna entropia krzyżowa
sp_cce_probas = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False)
sp_cce_logits = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)

tf.print(
    'Rzadka CCE (dla Probas): {:.4f}'.format(
    sp_cce_probas(y_true=[2], y_pred=probas)),
    '(dla Logits): {:.4f}'.format(
    sp_cce_logits(y_true=[2], y_pred=logits)))


# ## Implementacja głębokiej sieci splotowej za pomocą biblioteki TensorFlow
# 
# ### Architektura wielowarstwowej sieci CNN





# ### Wczytywanie i wstępne przetwarzanie danych








## Zestaw danych MNIST

mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset(shuffle_files=False)
print(datasets.keys())
mnist_train_orig, mnist_test_orig = datasets['train'], datasets['test']




BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20




mnist_train = mnist_train_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32)/255.0, 
                  tf.cast(item['label'], tf.int32)))

mnist_test = mnist_test_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32)/255.0, 
                  tf.cast(item['label'], tf.int32)))

tf.random.set_seed(1)

mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE,
                                  reshuffle_each_iteration=False)

mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)


# ### Implementowanie sieci CNN za pomocą interfejsu Keras
# 
# #### Konfigurowanie warstw sieci splotowej w interfejsie Keras
# 
#  * **Conv2D:** `tf.keras.layers.Conv2D`
#    * `filters`
#    * `kernel_size`
#    * `strides`
#    * `padding`
#    
#    
#  * **MaxPool2D:** `tf.keras.layers.MaxPool2D`
#    * `pool_size`
#    * `strides`
#    * `padding`
#    
#    
#  * **Dropout** `tf.keras.layers.Dropout2D`
#    * `rate`

# ### Konstruowanie sieci splotowej za pomocą interfejsu Keras



model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    data_format='channels_last',
    name='splot_1', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), name='laczaca_1'))
    
model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    name='splot_2', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='laczaca_2'))




model.compute_output_shape(input_shape=(16, 28, 28, 1))




model.add(tf.keras.layers.Flatten())

model.compute_output_shape(input_shape=(16, 28, 28, 1))




model.add(tf.keras.layers.Dense(
    units=1024, name='wpp_1', 
    activation='relu'))

model.add(tf.keras.layers.Dropout(
    rate=0.5))
    
model.add(tf.keras.layers.Dense(
    units=10, name='wpp_2',
    activation='softmax'))




tf.random.set_seed(1)
model.build(input_shape=(None, 28, 28, 1))

model.compute_output_shape(input_shape=(16, 28, 28, 1))




model.summary()




model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']) # to samo, co `tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')`

history = model.fit(mnist_train, epochs=NUM_EPOCHS, 
                    validation_data=mnist_valid, 
                    shuffle=True)




hist = history.history
x_arr = np.arange(len(hist['loss'])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='F. straty (uczenie)')
ax.plot(x_arr, hist['val_loss'], '--<', label='F. straty (walidacja)')
ax.set_xlabel('Epoka', size=15)
ax.set_ylabel('Funkcja straty', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Dokładność uczenia')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Dokładność walidacji')
ax.legend(fontsize=15)
ax.set_xlabel('Epoka', size=15)
ax.set_ylabel('Dokładność', size=15)

#plt.savefig('rysunki/15_12.png', dpi=300)
plt.show()




test_results = model.evaluate(mnist_test.batch(20))
print('\nDokładność dla danych testowych {:.2f}%'.format(test_results[1]*100))




batch_test = next(iter(mnist_test.batch(12)))

preds = model(batch_test[0])

tf.print(preds.shape)
preds = tf.argmax(preds, axis=1)
print(preds)

fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    img = batch_test[0][i, :, :, 0]
    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, '{}'.format(preds[i]), 
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center', 
            transform=ax.transAxes)
    
#plt.savefig('rysunki/15_13.png', dpi=300)
plt.show()





if not os.path.exists('modele'):
    os.mkdir('modele')


model.save('modele/mnist-cnn.h5')


# ----
# 
# Czytelnicy mogą zignorować poniższą komórkę.




