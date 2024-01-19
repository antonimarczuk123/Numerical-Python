#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Rozdział 13. Równoległe przetwarzanie sieci neuronowych za pomocą biblioteki TensorFlow (1/2)
# 

# Zwróć uwagę, że rozszerzenie zawierające nieobowiązkowy znak wodny stanowi niewielki plugin notatnika IPython / Jupyter, który zaprojektowałem w celu powielania kodu źródłowego. Wystarczy pominąć poniższe wiersze kodu:

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Sebastian Raschka & Vahid Mirjalili" -u -d -p numpy,scipy,matplotlib,tensorflow')


# In[2]:


from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Biblioteka TensorFlow a skuteczność uczenia

# ### Wyzwania związane z wydajnością

# In[3]:


Image(filename='rysunki/01.png', width=800)


# ### Czym jest biblioteka TensorFlow?

# In[4]:


Image(filename='rysunki/02.png', width=500)


# ### W jaki sposób będziemy poznawać bibliotekę TensorFlow?

# ## Pierwsze kroki z biblioteką TensorFlow

# ### Instalacja modułu TensorFlow

# In[5]:


#! pip install tensorflow


# In[6]:


import tensorflow as tf
print('Wersja modułu TensorFlow:', tf.__version__)
import numpy as np

np.set_printoptions(precision=3)


# In[7]:


get_ipython().system(" python -c 'import tensorflow as tf; print(tf.__version__)'")


# ### Tworzenie tensorów w TensorFlow

# In[8]:


a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]

t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

print(t_a)
print(t_b)


# In[9]:


tf.is_tensor(a), tf.is_tensor(t_a)


# In[10]:


t_ones = tf.ones((2, 3))

t_ones.shape


# In[11]:


t_ones.numpy()


# In[12]:


const_tensor = tf.constant([1.2, 5, np.pi], dtype=tf.float32)

print(const_tensor)


# ### Manipulowanie typem danych i rozmiarem tensora

# In[13]:


t_a_new = tf.cast(t_a, tf.int64)

print(t_a_new.dtype)


# In[14]:


t = tf.random.uniform(shape=(3, 5))

t_tr = tf.transpose(t)
print(t.shape, ' --> ', t_tr.shape)


# In[15]:


t = tf.zeros((30,))

t_reshape = tf.reshape(t, shape=(5, 6))

print(t_reshape.shape)


# In[16]:


t = tf.zeros((1, 2, 1, 4, 1))

t_sqz = tf.squeeze(t, axis=(2, 4))

print(t.shape, ' --> ', t_sqz.shape)


# ### Przeprowadzanie operacji matematycznych na tensorach

# In[17]:


tf.random.set_seed(1)

t1 = tf.random.uniform(shape=(5, 2), 
                       minval=-1.0,
                       maxval=1.0)

t2 = tf.random.normal(shape=(5, 2), 
                      mean=0.0,
                      stddev=1.0)


# In[18]:


t3 = tf.multiply(t1, t2).numpy()
print(t3)


# In[19]:


t4 = tf.math.reduce_mean(t1, axis=0)

print(t4)


# In[20]:


t5 = tf.linalg.matmul(t1, t2, transpose_b=True)

print(t5.numpy())


# In[21]:


t6 = tf.linalg.matmul(t1, t2, transpose_a=True)

print(t6.numpy())


# In[22]:


norm_t1 = tf.norm(t1, ord=2, axis=1).numpy()

print(norm_t1)


# In[23]:


np.sqrt(np.sum(np.square(t1), axis=1))


# ### Dzielenie, nawarstwianie i łączenie tensorów

# In[24]:


tf.random.set_seed(1)

t = tf.random.uniform((6,))

print(t.numpy())

t_splits = tf.split(t, 3)

[item.numpy() for item in t_splits]


# In[25]:


tf.random.set_seed(1)
t = tf.random.uniform((5,))

print(t.numpy())

t_splits = tf.split(t, num_or_size_splits=[3, 2])

[item.numpy() for item in t_splits]


# In[26]:


A = tf.ones((3,))
B = tf.zeros((2,))

C = tf.concat([A, B], axis=0)
print(C.numpy())


# In[27]:


A = tf.ones((3,))
B = tf.zeros((3,))

S = tf.stack([A, B], axis=1)
print(S.numpy())


# ## Tworzenie potoków wejściowych za pomocą tf.data, czyli interfejsu danych TensorFlow

# ### Tworzenie obiektów Dataset z istniejących tensorów

# In[28]:


a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]

ds = tf.data.Dataset.from_tensor_slices(a)

print(ds)


# In[29]:


for item in ds:
    print(item)


# In[30]:


ds_batch = ds.batch(3)

for i, elem in enumerate(ds_batch, 1):
    print('Grupa numer {}:'.format(i), elem.numpy())


# ### Łączenie dwóch tensorów we wspólny zestaw danych

# In[31]:


tf.random.set_seed(1)

t_x = tf.random.uniform([4, 3], dtype=tf.float32)
t_y = tf.range(4)


# In[32]:


ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)
    
ds_joint = tf.data.Dataset.zip((ds_x, ds_y))

for example in ds_joint:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())


# In[33]:


## method 2:
ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))

for example in ds_joint:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())


# In[34]:


ds_trans = ds_joint.map(lambda x, y: (x*2-1.0, y))

for example in ds_trans:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())


# ### Potasuj, pogrupuj, powtórz

# In[35]:


tf.random.set_seed(1)
ds = ds_joint.shuffle(buffer_size=len(t_x))

for example in ds:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())


# In[36]:


ds = ds_joint.batch(batch_size=3,
                    drop_remainder=False)

batch_x, batch_y = next(iter(ds))

print('Grupa-x: \n', batch_x.numpy())

print('Grupa-y:   ', batch_y.numpy())


# In[37]:


ds = ds_joint.batch(3).repeat(count=2)

for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())


# In[38]:


ds = ds_joint.repeat(count=2).batch(3)

for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())


# In[39]:


tf.random.set_seed(1)

## Order 1: shuffle -> batch -> repeat
ds = ds_joint.shuffle(4).batch(2).repeat(3)

for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())


# In[40]:


tf.random.set_seed(1)

## Kolejność 1.: tasowanie -> tworzenie grup -> powtórzenie
ds = ds_joint.shuffle(4).batch(2).repeat(20)

for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())


# In[41]:


tf.random.set_seed(1)

## Kolejność 2.: tworzenie grup -> tasowanie -> powtórzenie
ds = ds_joint.batch(2).shuffle(4).repeat(3)

for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())


# In[42]:


tf.random.set_seed(1)

## Kolejność 2.: tworzenie grup -> tasowanie -> powtórzenie
ds = ds_joint.batch(2).shuffle(4).repeat(20)

for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())


# ### Tworzenie zestawu danych z plików umieszczonych w lokalnym magazynie dyskowym

# In[43]:


import pathlib

imgdir_path = pathlib.Path('obrazy_psy_koty')

file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])

print(file_list)


# In[44]:


import matplotlib.pyplot as plt
import os


fig = plt.figure(figsize=(10, 5))
for i,file in enumerate(file_list):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    print('Rozmiar obrazu: ', img.shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)
    
# plt.savefig('r13-psykoty-obrazy.pdf')
plt.tight_layout()
plt.show()


# In[45]:


labels = [1 if 'pies' in os.path.basename(file) else 0
          for file in file_list]
print(labels)


# In[46]:


ds_files_labels = tf.data.Dataset.from_tensor_slices(
    (file_list, labels))

for item in ds_files_labels:
    print(item[0].numpy(), item[1].numpy())


# In[47]:


def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0

    return image, label

img_width, img_height = 120, 80

ds_images_labels = ds_files_labels.map(load_and_preprocess)

fig = plt.figure(figsize=(10, 5))
for i,example in enumerate(ds_images_labels):
    print(example[0].shape, example[1].numpy())
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0])
    ax.set_title('{}'.format(example[1].numpy()), 
                 size=15)
    
plt.tight_layout()
#plt.savefig('r13-psykoty-zestawdanych.pdf')
plt.show()


# ### Pobieranie dostępnych zestawów danych z biblioteki tensorflow_datasets

# In[48]:


get_ipython().system(' pip install tensorflow-datasets')


# In[49]:


import tensorflow_datasets as tfds

print(len(tfds.list_builders()))
print(tfds.list_builders()[:5])


# In[50]:


## Po uruchomieniu tego kodu zostanie wyświetlona pełna lista:
tfds.list_builders()


# Wczytywanie zestawu danych CelebA:

# In[51]:


celeba_bldr = tfds.builder('celeb_a')

print(celeba_bldr.info.features)
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features.keys())
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features['image'])
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features['attributes'].keys())
print('\n', 30*"=", '\n')
print(celeba_bldr.info.citation)


# In[52]:


# Pobiera dane, przygotowuje je i zapisuje na dysku
celeba_bldr.download_and_prepare()


# In[53]:


# Wczytuje dane z dysku jako tf.data.Datasets
datasets = celeba_bldr.as_dataset(shuffle_files=False)

datasets.keys()


# In[54]:


#import tensorflow as tf
ds_train = datasets['train']
assert isinstance(ds_train, tf.data.Dataset)

example = next(iter(ds_train))
print(type(example))
print(example.keys())


# In[55]:


ds_train = ds_train.map(lambda item: 
     (item['image'], tf.cast(item['attributes']['Male'], tf.int32)))


# In[56]:


ds_train = ds_train.batch(18)
images, labels = next(iter(ds_train))

print(images.shape, labels)


# In[57]:


fig = plt.figure(figsize=(12, 8))
for i,(image,label) in enumerate(zip(images, labels)):
    ax = fig.add_subplot(3, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    ax.set_title('{}'.format(label), size=15)
    

plt.show()


# Alternatywne sposoby wczytywania zestawów danych:

# In[58]:


mnist, mnist_info = tfds.load('mnist', with_info=True,
                              shuffle_files=False)

print(mnist_info)

print(mnist.keys())


# In[59]:


ds_train = mnist['train']

assert isinstance(ds_train, tf.data.Dataset)

ds_train = ds_train.map(lambda item: 
     (item['image'], item['label']))

ds_train = ds_train.batch(10)
batch = next(iter(ds_train))
print(batch[0].shape, batch[1])

fig = plt.figure(figsize=(15, 6))
for i,(image,label) in enumerate(zip(batch[0], batch[1])):
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image[:, :, 0], cmap='gray_r')
    ax.set_title('{}'.format(label), size=15)
    
plt.show()


# <br>
# <br>
# <br>

# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.

# In[60]:


get_ipython().system(' python ../.convert_notebook_to_script.py --input r13_cz1.ipynb --output r13_cz1.py')


# In[ ]:




