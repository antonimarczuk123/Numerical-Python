#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Python. Uczenie maszynowe - kod źródłowy

# # Rozdział 6. Najlepsze metody oceny modelu i strojenie parametryczne

# Zwróć uwagę, że rozszerzenie zawierające nieobowiązkowy znak wodny stanowi niewielki plugin notatnika IPython / Jupyter, który zaprojektowałem w celu powielania kodu źródłowego. Wystarczy pominąć poniższe wiersze kodu:

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Sebastian Raschka" -u -d -v -p numpy,pandas,matplotlib,sklearn')


# *Korzystanie z rozszerzenia `watermark` nie jest obowiązkowe. Możesz je zainstalować za pomocą polecenia:*  
# 
#     conda install watermark -c conda-forge  
# 
# lub  
# 
#     pip install watermark   
# 
# *Więcej informacji znajdziesz pod adresem: https://github.com/rasbt/watermark.*

# <br>
# <br>

# ### Spis treści

# - [Usprawnianie cyklu pracy za pomocą kolejkowania](#Usprawnianie-cyklu-pracy-za-pomocą-kolejkowania)
#   - [Wczytanie zestawu danych Breast Cancer Wisconsin](#Wczytanie-zestawu-danych-Breast-Cancer-Wisconsin)
#   - [Łączenie funkcji transformujących i estymatorów w kolejce czynności](#Łączenie-funkcji-transformujących-i-estymatorów-w-kolejce-czynności)
# - [Stosowanie k-krotnego sprawdzianu krzyżowego w ocenie skuteczności modelu](#Stosowanie-k-krotnego-sprawdzianu-krzyżowego-w-ocenie-skuteczności-modelu)
#   - [Metoda wydzielania](#Metoda-wydzielania)
#   - [K-krotny sprawdzian krzyżowy](#K-krotny-sprawdzian-krzyżowy)
# - [Sprawdzanie algorytmów za pomocą krzywych uczenia i krzywych walidacji](#Sprawdzanie-algorytmów-za-pomocą-krzywych-uczenia-i-krzywych-walidacji)
#   - [Diagnozowanie problemów z obciążeniem i wariancją za pomocą krzywych uczenia](#Diagnozowanie-problemów-z-obciążeniem-i-wariancją-za-pomocą-krzywych-uczenia)
#   - [Rozwiązywanie problemów przetrenowania i niedotrenowania za pomocą krzywych walidacji](#Rozwiązywanie-problemów-przetrenowania-i-niedotrenowania-za-pomocą-krzywych-walidacji)
# - [Dostrajanie modeli uczenia maszynowego za pomocą metody przeszukiwania siatki](#Dostrajanie-modeli-uczenia-maszynowego-za-pomocą-metody-przeszukiwania-siatki)
#   - [Strojenie hiperparametrów przy użyciu metody przeszukiwania siatki](#Strojenie-hiperparametrów-przy-użyciu-metody-przeszukiwania-siatki)
#   - [Dobór algorytmu poprzez zagnieżdżony sprawdzian krzyżowy](#Dobór-algorytmu-poprzez-zagnieżdżony-sprawdzian-krzyżowy)
# - [Przegląd wskaźników oceny skuteczności](#Przegląd-wskaźników-oceny-skuteczności)
#   - [Odczytywanie macierzy pomyłek](#Odczytywanie-macierzy-pomyłek)
#   - [Optymalizacja precyzji i pełności modelu klasyfikującego](#Optymalizacja-precyzji-i-pełności-modelu-klasyfikującego)
#   - [Wykres krzywej ROC](#Wykres-krzywej-ROC)
#   - [Wskaźniki zliczające dla klasyfikacji wieloklasowej](#Wskaźniki-zliczające-dla-klasyfikacji-wieloklasowej)
# - [Kwestia dysproporcji klas](#Kwestia-dysproporcji-klas)
# - [Podsumowanie](#Podsumowanie)

# <br>
# <br>

# In[2]:


from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# # Usprawnianie cyklu pracy za pomocą kolejkowania

# ...

# ## Wczytanie zestawu danych Breast Cancer Wisconsin

# In[3]:


import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)

# jeżeli zestaw danych Breast Cancer będzie tymczasowo niedostępny w
# repozytorium UCI, usuń znak komentarza z poniższego wiersza, aby wczytać ten
# zestaw z katalogu lokalnego:

# df = pd.read_csv('wdbc.data', header=None)

df.head()


# In[4]:


df.shape


# <hr>

# In[5]:


from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_


# In[6]:


le.transform(['M', 'B'])


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =     train_test_split(X, y, 
                     test_size=0.20,
                     stratify=y,
                     random_state=1)


# <br>
# <br>

# ## Łączenie funkcji transformujących i estymatorów w kolejce czynności

# In[8]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Dokładność dla danych testowych: %.3f' % pipe_lr.score(X_test, y_test))


# In[9]:


Image(filename='rysunki/06_01.png', width=500) 


# <br>
# <br>

# # Stosowanie k-krotnego sprawdzianu krzyżowego w ocenie skuteczności modelu

# ...

# ## Metoda wydzielania

# In[10]:


Image(filename='rysunki/06_02.png', width=500) 


# <br>
# <br>

# ## K-krotny sprawdzian krzyżowy

# In[11]:


Image(filename='rysunki/06_03.png', width=500) 


# In[12]:


import numpy as np
from sklearn.model_selection import StratifiedKFold
    

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Podzbiór: %2d, Rozkład klasy: %s, Dokładność: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nDokładność sprawdzianu: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[13]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('Wyniki dokładności sprawdzianu: %s' % scores)
print('Dokładność sprawdzianu: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# <br>
# <br>

# # Sprawdzanie algorytmów za pomocą krzywych uczenia i krzywych walidacji

# <br>
# <br>

# ## Diagnozowanie problemów z obciążeniem i wariancją za pomocą krzywych uczenia

# In[14]:


Image(filename='rysunki/06_04.png', width=600) 


# In[15]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', random_state=1,
                                           solver='lbfgs', max_iter=10000))

train_sizes, train_scores, test_scores =                learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Dokładność uczenia')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Dokładność walidacji')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Liczba przykładów uczących')
plt.ylabel('Dokładność')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
# plt.savefig('rysunki/06_05.png', dpi=300)
plt.show()


# <br>
# <br>

# ## Rozwiązywanie problemów przetrenowania i niedotrenowania za pomocą krzywych walidacji

# In[16]:


from sklearn.model_selection import validation_curve


param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                param_name='logisticregression__C', 
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='Dokładność uczenia')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='Dokładność walidacji')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parametr C')
plt.ylabel('Dokładność')
plt.ylim([0.8, 1.0])
plt.tight_layout()
# plt.savefig('rysunki/06_06.png', dpi=300)
plt.show()


# <br>
# <br>

# # Dostrajanie modeli uczenia maszynowego za pomocą metody przeszukiwania siatki

# <br>
# <br>

# ## Strojenie hiperparametrów przy użyciu metody przeszukiwania siatki

# In[17]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# In[18]:


clf = gs.best_estimator_

# clf.fit(X_train, y_train) 
# zwróć uwagę, że nie musimy ponownie dopasowywać klasyfikatora,
# ponieważ proces ten jest realizowany automatycznie za pomocą argumentu refit=True.

print('Dokładność dla zbioru testowego: %.3f' % clf.score(X_test, y_test))


# <br>
# <br>

# ## Dobór algorytmu poprzez zagnieżdżony sprawdzian krzyżowy

# In[19]:


Image(filename='rysunki/06_07.png', width=500) 


# In[20]:


gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)
print('Dokładność sprawdzianu krzyżowego: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))


# In[21]:


from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)
print('Dokładność sprawdzianu krzyżowego: %.3f +/- %.3f' % (np.mean(scores), 
                                      np.std(scores)))


# <br>
# <br>

# # Przegląd wskaźników oceny skuteczności

# ...

# ## Odczytywanie macierzy pomyłek

# In[22]:


Image(filename='rysunki/06_08.png', width=300) 


# In[23]:


from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# In[24]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Przewidywana etykieta')
plt.ylabel('Rzeczywista etykieta')

plt.tight_layout()
#plt.savefig('rysunki/06_09.png', dpi=300)
plt.show()


# ### Dodatkowa uwaga

# Przypominam, że na początku rozdziału zakodowaliśmy etykiety klas w taki sposób, że przykłady *złośliwe* należą do klasy "pozytywnej" (1), a przykłady *łagodne* do klasy "negatywnej" (0):

# In[25]:


le.transform(['M', 'B'])


# In[26]:


confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# Następnie wyświetliliśmy macierz pomyłek w następujący sposób:

# In[27]:


confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# Zwróć uwagę, że (prawdziwe) przykłady klasy 0 poprawnie sklasyfikowane jako klasa 0 (prawdziwie negatywne) znajdują się teraz w lewym górnym rogu macierzy (indeks 0,0). Aby zmienić kolejność wyświetlania komórek, tak aby prawdziwie negatywne znajdowały się w prawym dolnym rogu tabeli (indeks 1,1), a prawdziwie pozytywne w jej lewej górnej części, możemy użyć argumentu `labels`:

# In[28]:


confmat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[1, 0])
print(confmat)


# Podsumowując:
# 
# Zakładając w tym przykładzie, że klasa 1 (nowotwory złośliwe) jest pozytywna, nasz model poprawnie sklasyfikował 71 przykładów należących do klasy 0 (prawdziwie negatywne) oraz 40 przykładów do klasy 1 (prawdziwie pozytywne). Jednocześnie jednak jeden przykład z klasy 0 został nieprawidłowo sklasyfikowany do klasy 1 (fałszywie pozytywny) oraz model przewidział, że dwa przykłady są łagodne, pomimo że w rzeczywistości reprezentują nowotwór złośliwy (fałszywie negatywne).

# <br>
# <br>

# ## Optymalizacja precyzji i pełności modelu klasyfikującego

# In[29]:


from sklearn.metrics import precision_score, recall_score, f1_score

print('Precyzja: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Pełność: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))


# In[30]:


from sklearn.metrics import make_scorer

scorer = make_scorer(f1_score, pos_label=0)

c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# <br>
# <br>

# ## Wykres krzywej ROC

# In[31]:


from sklearn.metrics import roc_curve, auc
from distutils.version import LooseVersion as Version
from scipy import __version__ as scipy_version


if scipy_version >= Version('1.4.1'):
    from numpy import interp
else:
    from scipy import interp


pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2', 
                                           random_state=1,
                                           solver='lbfgs',
                                           C=100.0))

X_train2 = X_train[:, [4, 14]]
    

cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label='Podzbiór nr %d (obszar = %0.2f)'
                   % (i+1, roc_auc))

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='Losowe zgadywanie')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Uśredniona krzywa ROC (obszar = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='Doskonała skuteczność')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Odsetek fałszywie pozytywnych')
plt.ylabel('Odsetek prawdziwie pozytywnych')
plt.legend(loc="lower right")

plt.tight_layout()
# plt.savefig('rysunki/06_10.png', dpi=300)
plt.show()


# <br>
# <br>

# ## Wskaźniki zliczające dla klasyfikacji wieloklasowej

# In[32]:


pre_scorer = make_scorer(score_func=precision_score, 
                         pos_label=1, 
                         greater_is_better=True, 
                         average='micro')


# # Kwestia dysproporcji klas

# In[33]:


X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))


# In[34]:


y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100


# In[35]:


from sklearn.utils import resample

print('Liczba przykładów z klasy 1. przed przepróbkowaniem:', X_imb[y_imb == 1].shape[0])

X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)

print('Liczba przykładów z klasy 1. po przepróbkowaniu:', X_upsampled.shape[0])


# In[36]:


X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))


# In[37]:


y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100


# <br>
# <br>

# # Podsumowanie

# ...

# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.

# In[38]:


get_ipython().system(' python ../.convert_notebook_to_script.py --input r06.ipynb --output r06.py')

