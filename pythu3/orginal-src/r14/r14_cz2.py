# coding: utf-8


import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn
import sklearn.model_selection

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Rozdział 14. Czas na szczegóły — mechanizm działania biblioteki TensorFlow (2/3)

# Zwróć uwagę, że rozszerzenie zawierające nieobowiązkowy znak wodny stanowi niewielki plugin notatnika IPython / Jupyter, który zaprojektowałem w celu powielania kodu źródłowego. Wystarczy pominąć poniższe wiersze kodu:










# ## Estymatory TensorFlow
# 
# ##### Etapy korzystania z gotowych estymatorów
# 
#  * **Etap 1.** Zdefiniuj funkcję wejściową importującą dane   
#  * **Etap 2.** Zdefiniuje kolumny cech łączące estymator z danymi   
#  * **etap 3.** Utwórz wystąpienie estymatora lub przekształć model Keras w estymator   
#  * **Etap 4.** Użyj estymatora: train() - evaluate() - predict()   



tf.random.set_seed(1)
np.random.seed(1)


# ### Praca z kolumnami cech
# 
# 
#  * Definicja: https://developers.google.com/machine-learning/glossary/#feature_columns
#  * Dokumentacja: https://www.tensorflow.org/api_docs/python/tf/feature_column







dataset_path = tf.keras.utils.get_file("auto-mpg.data", 
                                       ("http://archive.ics.uci.edu/ml/machine-learning-databases"
                                        "/auto-mpg/auto-mpg.data"))

column_names = ['Spalanie', 'Cylindry', 'Poj_skokowa', 'Moc',
                'Waga', 'Przyspieszenie', 'Rocznik_modelu', 'Obszar_pochodzenia']

df = pd.read_csv(dataset_path, names=column_names,
                 na_values = "?", comment='\t',
                 sep=" ", skipinitialspace=True)

df.tail()




print(df.isna().sum())

df = df.dropna()
df = df.reset_index(drop=True)
df.tail()






df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=0.8)
train_stats = df_train.describe().transpose()
train_stats




numeric_column_names = ['Cylindry', 'Poj_skokowa', 'Moc', 'Waga', 'Przyspieszenie']

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()

for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std  = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean)/std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean)/std
    
df_train_norm.tail()


# #### Kolumny numeryczne



numeric_features = []

for col_name in numeric_column_names:
    numeric_features.append(tf.feature_column.numeric_column(key=col_name))
    
numeric_features




feature_year = tf.feature_column.numeric_column(key="Rocznik_modelu")

bucketized_features = []

bucketized_features.append(tf.feature_column.bucketized_column(
    source_column=feature_year,
    boundaries=[73, 76, 79]))

print(bucketized_features)




feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Obszar_pochodzenia',
    vocabulary_list=[1, 2, 3])

categorical_indicator_features = []
categorical_indicator_features.append(tf.feature_column.indicator_column(feature_origin))

print(categorical_indicator_features)


# ### Uczenie maszynowe za pomocą gotowych estymatorów



def train_input_fn(df_train, batch_size=8):
    df = df_train.copy()
    train_x, train_y = df, df.pop('Spalanie')
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))

    # tasuje, powtarza i tworzy grupy przykładów
    return dataset.shuffle(1000).repeat().batch(batch_size)

## sprawdzenie
ds = train_input_fn(df_train_norm)
batch = next(iter(ds))
print('Klucze:', batch[0].keys())
print('Grupa (Rocznik_modelu):', batch[0]['Rocznik_modelu'])




all_feature_columns = (numeric_features + 
                       bucketized_features + 
                       categorical_indicator_features)

print(all_feature_columns)




regressor = tf.estimator.DNNRegressor(
    feature_columns=all_feature_columns,
    hidden_units=[32, 10],
    model_dir='modele/autompg-dnnregresor/')




EPOCHS = 1000
BATCH_SIZE = 8
total_steps = EPOCHS * int(np.ceil(len(df_train) / BATCH_SIZE))
print('Kroki uczenia:', total_steps)

regressor.train(
    input_fn=lambda:train_input_fn(df_train_norm, batch_size=BATCH_SIZE),
    steps=total_steps)




reloaded_regressor = tf.estimator.DNNRegressor(
    feature_columns=all_feature_columns,
    hidden_units=[32, 10],
    warm_start_from='modele/autompg-dnnregresor/',
    model_dir='modele/autompg-dnnregresor/')




def eval_input_fn(df_test, batch_size=8):
    df = df_test.copy()
    test_x, test_y = df, df.pop('Spalanie')
    dataset = tf.data.Dataset.from_tensor_slices((dict(test_x), test_y))

    return dataset.batch(batch_size)

eval_results = reloaded_regressor.evaluate(
    input_fn=lambda:eval_input_fn(df_test_norm, batch_size=8))

for key in eval_results:
    print('{:15s} {}'.format(key, eval_results[key]))
    
print('Średnia funkcja straty {:.4f}'.format(eval_results['average_loss']))




pred_res = regressor.predict(input_fn=lambda: eval_input_fn(df_test_norm, batch_size=8))

print(next(iter(pred_res)))


# #### Regresor wzmocnionego drzewa



boosted_tree = tf.estimator.BoostedTreesRegressor(
    feature_columns=all_feature_columns,
    n_batches_per_layer=20,
    n_trees=200)

boosted_tree.train(
    input_fn=lambda:train_input_fn(df_train_norm, batch_size=BATCH_SIZE))

eval_results = boosted_tree.evaluate(
    input_fn=lambda:eval_input_fn(df_test_norm, batch_size=8))

print(eval_results)

print('Średnia funkcja straty {:.4f}'.format(eval_results['average_loss']))


# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.




