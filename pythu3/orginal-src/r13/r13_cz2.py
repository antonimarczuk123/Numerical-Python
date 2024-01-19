# coding: utf-8


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from scipy.special import expit

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Rozdział 13. Równoległe przetwarzanie sieci neuronowych za pomocą biblioteki TensorFlow (2/2)
# 

# Zwróć uwagę, że rozszerzenie zawierające nieobowiązkowy znak wodny stanowi niewielki plugin notatnika IPython / Jupyter, który zaprojektowałem w celu powielania kodu źródłowego. Wystarczy pominąć poniższe wiersze kodu:









# ## Tworzenie modelu sieci neuronowej za pomocą modułu TensorFlow

# ### Interfejs Keras (tf.keras)

# ### Tworzenie modelu regresji liniowej







X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1,
                    2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0,
                    9.0])


plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()




X_train_norm = (X_train - np.mean(X_train))/np.std(X_train)

ds_train_orig = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_train_norm, tf.float32),
     tf.cast(y_train, tf.float32)))




class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = tf.Variable(0.0, name='waga')
        self.b = tf.Variable(0.0, name='obciazenie')

    def call(self, x):
        return self.w*x + self.b


model = MyModel()

model.build(input_shape=(None, 1))
model.summary()




def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


## testowanie funkcji:
yt = tf.convert_to_tensor([1.0])
yp = tf.convert_to_tensor([1.5])

loss_fn(yt, yp)




def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dW, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)




tf.random.set_seed(1)

num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epoch = int(np.ceil(len(y_train) / batch_size))


ds_train = ds_train_orig.shuffle(buffer_size=len(y_train))
ds_train = ds_train.repeat(count=None)
ds_train = ds_train.batch(1)

Ws, bs = [], []

for i, batch in enumerate(ds_train):
    if i >= steps_per_epoch * num_epochs:
        break
    Ws.append(model.w.numpy())
    bs.append(model.b.numpy())

    bx, by = batch
    loss_val = loss_fn(model(bx), by)

    train(model, bx, by, learning_rate=learning_rate)
    if i%log_steps==0:
        print('Epoka numer {:4d} Krok numer  {:2d} Funkcja straty {:6.4f}'.format(
              int(i/steps_per_epoch), i, loss_val))




print('Parametry końcowe:', model.w.numpy(), model.b.numpy())


X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)

y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))


fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Przykłady uczące', 'Reg. liniowa'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(Ws, lw=3)
plt.plot(bs, lw=3)
plt.legend(['Waga w', 'Obc. jednostkowe b'], fontsize=15)
ax.set_xlabel('Iteracja', size=15)
ax.set_ylabel('Wartość', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
#plt.savefig('r13-reglin-1.pdf')

plt.show()


# ### Uczenie modelu za pomocą metod .compile() i .fit()



tf.random.set_seed(1)
model = MyModel()
#model.build((None, 1))

model.compile(optimizer='sgd', 
              loss=loss_fn,
              metrics=['mae', 'mse'])




model.fit(X_train_norm, y_train, 
          epochs=num_epochs, batch_size=batch_size,
          verbose=1)




print(model.w.numpy(), model.b.numpy())


X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)

y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))


fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Przykłady uczące', 'Regresja liniowa'], fontsize=15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(Ws, lw=3)
plt.plot(bs, lw=3)
plt.legend(['W', 'Obciążenie'], fontsize=15)
plt.show()


# ## Tworzenie perceptronu wielowarstwowego klasyfikującego kwiaty z zestawu danych Iris






iris, iris_info = tfds.load('iris', with_info=True)

print(iris_info)




tf.random.set_seed(1)

ds_orig = iris['train']
ds_orig = ds_orig.shuffle(150, reshuffle_each_iteration=False)

print(next(iter(ds_orig)))

ds_train_orig = ds_orig.take(100)
ds_test = ds_orig.skip(100)




## sprawdza liczbę przykładów:

n = 0
for example in ds_train_orig:
    n += 1
print(n)


n = 0
for example in ds_test:
    n += 1
print(n)




ds_train_orig = ds_train_orig.map(
    lambda x: (x['features'], x['label']))

ds_test = ds_test.map(
    lambda x: (x['features'], x['label']))

next(iter(ds_train_orig))




iris_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='sigmoid', 
                          name='fc1', input_shape=(4,)),
    tf.keras.layers.Dense(3, name='fc2', activation='softmax')])

iris_model.summary()




iris_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])




num_epochs = 100
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil(training_size / batch_size)

ds_train = ds_train_orig.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000)


history = iris_model.fit(ds_train, epochs=num_epochs,
                         steps_per_epoch=steps_per_epoch, 
                         verbose=0)




hist = history.history

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(hist['loss'], lw=3)
ax.set_title('Funkcja straty uczenia', size=15)
ax.set_xlabel('Epoka', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Dokładność uczenia', size=15)
ax.set_xlabel('Epoka', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
#plt.savefig('r13-cls-krzywa-uczenia.pdf')

plt.show()


# ### Ocena wytrenowanego modelu za pomocą danych testowych



results = iris_model.evaluate(ds_test.batch(50), verbose=0)
print('F. straty (test): {:.4f}   Dokładność (test): {:.4f}'.format(*results))


# ### Zapisywanie i wczytywanie wyuczonego modelu



iris_model.save('iris-klasyfikator.h5', 
                overwrite=True,
                include_optimizer=True,
                save_format='h5')




iris_model_new = tf.keras.models.load_model('iris-klasyfikator.h5')

iris_model_new.summary()




results = iris_model_new.evaluate(ds_test.batch(50), verbose=0)
print('F. straty (test): {:.4f}   Dokładność (test): {:.4f}'.format(*results))




labels_train = []
for i,item in enumerate(ds_train_orig):
    labels_train.append(item[1].numpy())
    
labels_test = []
for i,item in enumerate(ds_test):
    labels_test.append(item[1].numpy())
print('Zbiór uczący: ',len(labels_train), 'Zbiór testowy: ', len(labels_test))




iris_model_new.to_json()


# ## Dobór funkcji aktywacji dla wielowarstwowych sieci neuronowych
# 

# ### Funkcja logistyczna — powtórzenie




X = np.array([1, 1.4, 2.5]) ## pierwsza wartość musi być równa 1
w = np.array([0.4, 0.3, 0.5])

def net_input(X, w):
    return np.dot(X, w)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

print('P(y=1|x) = %.3f' % logistic_activation(X, w)) 




# W : tablica, wymiary = [n_jednostek_wyjściowych, n_jednostek_ukrytych+1]
#   zwróć uwagę, że pierwsza kolumna zawiera jednostki obciążenia

W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])

# A : tablica, wymiary = [n_jednostek_ukrytych+1, n_przykładów]
#    zwróć uwagę, że pierwsza kolumna w tej tablicy musi być równa 1


A = np.array([[1, 0.1, 0.4, 0.6]])
Z = np.dot(W, A[0])
y_probas = logistic(Z)
print('Pobudzenie całkowite: \n', Z)

print('Jednostki wyjściowe:\n', y_probas) 




y_class = np.argmax(Z, axis=0)
print('Przewidywana etykieta klas: %d' % y_class) 


# ### Szacowanie prawdopodobieństw przynależności do klas w klasyfikacji wieloklasowej za pomocą funkcji softmax



def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

y_probas = softmax(Z)
print('Prawdopodobieństwa:\n', y_probas)

np.sum(y_probas)





Z_tensor = tf.expand_dims(Z, axis=0)
tf.keras.activations.softmax(Z_tensor)


# ### Rozszerzanie zakresu wartości wyjściowych za pomocą funkcji tangensa hiperbolicznego




def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.ylim([-1.5, 1.5])
plt.xlabel('Pobudzenie całkowite $z$')
plt.ylabel('Aktywacja $\phi(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')
plt.plot(z, tanh_act,
    linewidth=3, linestyle='--',
    label='Funkcja tanh')
plt.plot(z, log_act,
    linewidth=3,
    label='Funkcja logistyczna')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()




np.tanh(z)





tf.keras.activations.tanh(z)





expit(z)




tf.keras.activations.sigmoid(z)


# ### Aktywacja za pomocą prostowanej jednostki liniowej (ReLU)




tf.keras.activations.relu(z)


# ## Podsumowanie

# # Dodatek
# 
# ## Rozdzielanie zestawu danych: ryzyko wymieszania przykładów uczących z testowymi



## właściwy sposób:
ds = tf.data.Dataset.range(15)
ds = ds.shuffle(15, reshuffle_each_iteration=False)


ds_train = ds.take(10)
ds_test = ds.skip(10)

ds_train = ds_train.shuffle(10).repeat(10)
ds_test = ds_test.shuffle(5)
ds_test = ds_test.repeat(10)

set_train = set()
for i,item in enumerate(ds_train):
    set_train.add(item.numpy())

set_test = set()
for i,item in enumerate(ds_test):
    set_test.add(item.numpy())

print(set_train, set_test)




## niewłaściwy sposób:
ds = tf.data.Dataset.range(15)
ds = ds.shuffle(15, reshuffle_each_iteration=True)


ds_train = ds.take(10)
ds_test = ds.skip(10)

ds_train = ds_train.shuffle(10).repeat(10)
ds_test = ds_test.shuffle(5)
ds_test = ds_test.repeat(10)

set_train = set()
for i,item in enumerate(ds_train):
    set_train.add(item.numpy())

set_test = set()
for i,item in enumerate(ds_test):
    set_test.add(item.numpy())

print(set_train, set_test)


# ### Rozdzielanie zestawu danych za pomocą `tfds.Split`




##--------------------------- Ostrzeżenie ----------------------------##
##                                                                    ##
##     Uwaga: obecnie tfds.Split zawiera błąd w wersji TF 2.0.0       ##
##                                                                    ##
##  Np. podział [2, 1] powinien dać nam 100 przykładów                ##
##      uczących i 50 przykładów testowych.                           ##
##                                                                    ##
##  Zamiast tego otrzymujemy 116 przykładów uczących                  ##
##  i 34 przykłady testowe.                                           ##
##                                                                    ##
##--------------------------------------------------------------------##


##  Metoda 1. Wyznaczenie wartości procentowych
#first_67_percent = tfds.Split.TRAIN.subsplit(tfds.percent[:67])
#last_33_percent = tfds.Split.TRAIN.subsplit(tfds.percent[-33:])

#ds_train_orig = tfds.load('iris', split=first_67_percent)
#ds_test = tfds.load('iris', split=last_33_percent)


##  Metoda 2. Określenie wag
split_train, split_test = tfds.Split.TRAIN.subsplit([2, 1])

ds_train_orig = tfds.load('iris', split=split_train)
ds_test = tfds.load('iris', split=split_test)

print(next(iter(ds_train_orig)))
print()
print(next(iter(ds_test)))


ds_train_orig = ds_train_orig.shuffle(100, reshuffle_each_iteration=True)
ds_test  = ds_test.shuffle(50, reshuffle_each_iteration=False)

ds_train_orig = ds_train_orig.map(
    lambda x: (x['features'], x['label']))

ds_test = ds_test.map(
    lambda x: (x['features'], x['label']))

print(next(iter(ds_train_orig)))


for j in range(5):
    labels_train = []
    for i,item in enumerate(ds_train_orig):
        labels_train.append(item[1].numpy())

    labels_test = []
    for i,item in enumerate(ds_test):
        labels_test.append(item[1].numpy())
    print('Zbiór uczący: ',len(labels_train), 'Zbiór testowy: ', len(labels_test))

    labels_test = np.array(labels_test)

    print(np.sum(labels_test == 0), np.sum(labels_test == 1), np.sum(labels_test == 2))



# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.




