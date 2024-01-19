#!/usr/bin/env python
# coding: utf-8

# *Python Machine Learning 3rd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2019
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Python. Uczenie maszynowe - kod źródłowy

# # Rozdział 3. Stosowanie klasyfikatorów uczenia maszynowego za pomocą biblioteki scikit-learn

# Zwróć uwagę, że rozszerzenie zawierające nieobowiązkowy znak wodny stanowi niewielki plugin notatnika IPython / Jupyter, który zaprojektowałem w celu powielania kodu źródłowego. Wystarczy pominąć poniższe wiersze kodu:

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Sebastian Raschka" -u -d -p numpy,pandas,matplotlib,sklearn')


# *Korzystanie z rozszerzenia `watermark` nie jest obowiązkowe. Możesz je zainstalować za pomocą polecenia:*  
# 
#     conda install watermark -c conda-forge  
# 
# lub  
# 
#     pip install watermark   
# 
# *Więcej informacji znajdziesz pod adresem: https://github.com/rasbt/watermark.*

# ### Spis treści

# - [Wybór algorytmu klasyfikującego](#Wybór-algorytmu-klasyfikującego)
# - [Pierwsze kroki z biblioteką scikit-learn - uczenie perceptronu](#Pierwsze-kroki-z-biblioteką-scikit-learn-—-uczenie-perceptronu)
#     - [Uczenie perceptronu za pomocą biblioteki scikit-learn](#Uczenie-perceptronu-za-pomocą-biblioteki-scikit-learn)
# - [Modelowanie prawdopodobieństwa przynależności do klasy za pomocą regresji logistycznej](#Modelowanie-prawdopodobieństwa-przynależności-do-klasy-za-pomocą-regresji-logistycznej)
#     - [Teoretyczne podłoże regresji logistycznej i prawdopodobieństwa warunkowego](#Teoretyczne-podłoże-regresji-logistycznej-i-prawdopodobieństwa-warunkowego)
#     - [Wyznaczanie wag logistycznej funkcji kosztu](#Wyznaczanie-wag-logistycznej-funkcji-kosztu)
#     - [Uczenie modelu regresji logistycznej za pomocą biblioteki scikit-learn](#Uczenie-modelu-regresji-logistycznej-za-pomocą-biblioteki-scikit-learn)
#     - [Zapobieganie przetrenowaniu za pomocą regularyzacji](#Zapobieganie-przetrenowaniu-za-pomocą-regularyzacji)
# - [Wyznaczanie maksymalnego marginesu za pomocą maszyn wektorów nośnych](#Wyznaczanie-maksymalnego-marginesu-za-pomocą-maszyn-wektorów-nośnych)
#     - [Teoretyczne podłoże maksymalnego marginesu](#Teoretyczne-podłoże-maksymalnego-marginesu)
#     - [Rozwiązywanie przypadków nieliniowo rozdzielnych za pomocą zmiennych uzupełniających](#Rozwiązywanie-przypadków-nieliniowo-rozdzielnych-za-pomocą-zmiennych-uzupełniających)
#     - [Alternatywne implementacje w interfejsie scikit-learn](#Alternatywne-implementacje-w-interfejsie-scikit-learn)
# - [Rozwiązywanie nieliniowych problemów za pomocą jądra SVM](#Rozwiązywanie-nieliniowych-problemów-za-pomocą-jądra-SVM)
#     - [Stosowanie sztuczki z funkcją jądra do znajdowania przestrzeni rozdzielających w przestrzeni wielowymiarowej](#Stosowanie-sztuczki-z-funkcją-jądra-do-znajdowania-przestrzeni-rozdzielających-w-przestrzeni-wielowymiarowej)
# - [Uczenie drzew decyzyjnych](#Uczenie-drzew-decyzyjnych)
#     - [Maksymalizowanie przyrostu informacji — osiąganie jak największych korzyści](#Maksymalizowanie-przyrostu-informacji-—-osiąganie-jak-największych-korzyści)
#     - [Budowanie drzewa decyzyjnego](#Budowanie-drzewa-decyzyjnego)
#     - [Łączenie wielu drzew decyzyjnych za pomocą modelu losowego lasu](#Łączenie-wielu-drzew-decyzyjnych-za-pomocą-modelu-losowego-lasu)
# - [Algorytm k-najbliższych sąsiadów — model leniwego uczenia](#Algorytm-k-najbliższych-sąsiadów-—-model-leniwego-uczenia)
# - [Podsumowanie](#Podsumowanie)

# <br>
# <br>

# <br>
# <br>

# In[2]:


from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# # Wybór algorytmu klasyfikującego

# ...

# # Pierwsze kroki z biblioteką scikit-learn — uczenie perceptronu

# Wczytujemy zestaw danych Iris z biblioteki scikit-learn. Trzecia kolumna reprezentuje tutaj długość płatka, a czwarta - jego szerokość. Klasy są przekształcone do postaci liczb całkowitych, gdzie etykieta 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.

# In[3]:


from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Etykiety klas:', np.unique(y))


# Dzielimy dane na 70% przykładów uczących i 30% przykładów testowych:

# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)


# In[5]:


print('Liczba etykiet w zbiorze y:', np.bincount(y))
print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))


# Standaryzacja cech:

# In[6]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# <br>
# <br>

# ## Uczenie perceptronu za pomocą biblioteki scikit-learn

# In[7]:


from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)


# **Uwaga**
# 
# - Możesz zastąpić parametr `Perceptron(n_iter, ...)` parametrem `Perceptron(max_iter, ...)` w bibliotece scikit-learn >= 0.19. Celowo wykorzystujemy tu parametr `n_iter`, ponieważ wiele osób ciągle korzysta z wersji 0.18 biblioteki scikit-learn.

# In[8]:


y_pred = ppn.predict(X_test_std)
print('Nieprawidłowo sklasyfikowane przykłady: %d' % (y_test != y_pred).sum())


# In[9]:


from sklearn.metrics import accuracy_score

print('Dokładność: %.3f' % accuracy_score(y_test, y_pred))


# In[10]:


print('Dokładność: %.3f' % ppn.score(X_test_std, y_test))


# In[11]:


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # konfiguruje generator znaczników i mapę kolorów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # rysuje wykres powierzchni decyzyjnej
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # zaznacza przykłady testowe
    if test_idx:
        # umieszcza na wykresie wszystkie przykłady
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='Zestaw testowy')


# Trenowanie modelu perceptronu za pomocą standaryzowanych danych uczących:

# In[12]:


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('Długość płatka [standaryzowana]')
plt.ylabel('Szerokość płatka [standaryzowana]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('rysunki/03_01.png', dpi=300)
plt.show()


# <br>
# <br>

# # Modelowanie prawdopodobieństwa przynależności do klasy za pomocą regresji logistycznej

# ...

# ### Teoretyczne podłoże regresji logistycznej i prawdopodobieństwa warunkowego

# In[13]:


import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
#plt.savefig('rysunki/03_02.png', dpi=300)
plt.show()


# In[14]:


Image(filename='rysunki/03_03.png', width=500) 


# <br>
# <br>

# ### Wyznaczanie wag logistycznej funkcji kosztu

# In[15]:


def cost_1(z):
    return - np.log(sigmoid(z))


def cost_0(z):
    return - np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) jeżeli y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) jeżeli y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('rysunki/03_04.png', dpi=300)
plt.show()


# In[16]:


class LogisticRegressionGD(object):
    """Klasyfikator regresji logistycznej wykorzystujący metodę gradientu prostego

    Parametry
    ------------
    eta : zmiennoprzecinkowy
       Współczynnik uczenia (pomiędzy 0,0 a 1,0)
    n_iter : liczba całkowita
       Przebiegi po zestawie danych uczących.
    random_state : liczba całkowita
       Ziarno generatora liczb losowych do losowej inicjalizacji wag.

    Atrybuty
    -----------
    w_ : tablica jednowymiarowa
       Wagi po dopasowaniu.
    cost_ : lista
       Wartość funkcji kosztu (suma kwadratów) w każdej epoce


    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Dopasowanie danych uczących.

        Parametry
        ----------
        X : {tablicopodobny}, wymiary = [n_przykładów, n_cech]
           Wektory uczenia, gdzie n_przykładów oznacza liczbę przykładów, a
           n_cech — liczbę cech.
        y : tablicopodobny, wymiary = [n_przykładów]
           Wartości docelowe.

        Zwraca
        -------
        self : obiekt


        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            # zwróć uwagę, że obliczamy teraz `koszt` logistyczny,
            # a nie sumę kwadratów błędów
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Obliczanie pobudzenia całkowitego"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Obliczanie logistycznej, sigmoidalnej funkcji aktywacji"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Zwraca etykietę klasy po skoku jednostkowym"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # jest to równoważne wierszowi:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# <br>
# <br>

# In[17]:


X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('Długość płatka [standaryzowana]')
plt.ylabel('Szerokość płatka [standaryzowana]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('rysunki/03_05.png', dpi=300)
plt.show()


# ### Uczenie modelu regresji logistycznej za pomocą biblioteki scikit-learn

# In[18]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('Długość płatka [standaryzowana]')
plt.ylabel('Szerokość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('rysunki/03_06.png', dpi=300)
plt.show()


# In[19]:


lr.predict_proba(X_test_std[:3, :])


# In[20]:


lr.predict_proba(X_test_std[:3, :]).sum(axis=1)


# In[21]:


lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)


# In[22]:


lr.predict(X_test_std[:3, :])


# In[23]:


lr.predict(X_test_std[0, :].reshape(1, -1))


# <br>
# <br>

# ### Zapobieganie przetrenowaniu za pomocą regularyzacji

# In[24]:


Image(filename='rysunki/03_07.png', width=700) 


# In[25]:


weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=1,
                            solver='lbfgs',
                            multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label='Długość płatka')
plt.plot(params, weights[:, 1], linestyle='--',
         label='Szerokość płatka')
plt.ylabel('Współczynnik wag')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
#plt.savefig('rysunki/03_08.png', dpi=300)
plt.show()


# <br>
# <br>

# # Wyznaczanie maksymalnego marginesu za pomocą maszyn wektorów nośnych

# In[26]:


Image(filename='rysunki/03_09.png', width=700) 


# ## Teoretyczne podłoże maksymalnego marginesu

# ...

# ## Rozwiązywanie przypadków nieliniowo rozdzielnych za pomocą zmiennych uzupełniających

# In[27]:


Image(filename='rysunki/03_10.png', width=600) 


# In[28]:


from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))
plt.xlabel('Długość płatka [standaryzowana]')
plt.ylabel('Szerokość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('rysunki/03_11.png', dpi=300)
plt.show()


# ## Alternatywne implementacje w interfejsie scikit-learn

# In[29]:


from sklearn.linear_model import SGDClassifier

ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')


# <br>
# <br>

# # Rozwiązywanie nieliniowych problemów za pomocą jądra SVM

# In[30]:


import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('rysunki/03_12.png', dpi=300)
plt.show()


# In[31]:


Image(filename='rysunki/03_13.png', width=700) 


# <br>
# <br>

# ## Stosowanie sztuczki z funkcją jądra do znajdowania przestrzeni rozdzielających w przestrzeni wielowymiarowej

# In[32]:


svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('rysunki/03_14.png', dpi=300)
plt.show()


# In[33]:


from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('Długość płatka [standaryzowana]')
plt.ylabel('Szerokość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('rysunki/03_15.png', dpi=300)
plt.show()


# In[34]:


svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('Długość płatka [standaryzowana]')
plt.ylabel('Szerokość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('rysunki/03_16.png', dpi=300)
plt.show()


# <br>
# <br>

# # Uczenie drzew decyzyjnych

# In[35]:


Image(filename='rysunki/03_17.png', width=500) 


# In[36]:


Image(filename='rysunki/03_18.png', width=500) 


# <br>
# <br>

# ## Maksymalizowanie przyrostu informacji — osiąganie jak największych korzyści

# In[37]:


import matplotlib.pyplot as plt
import numpy as np


def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropia', 'Entropia (skalowana)', 
                           'Wskaźnik Giniego', 'Błąd klasyfikacji'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Wskaźnik zanieczyszczenia')
#plt.savefig('rysunki/03_19.png', dpi=300, bbox_inches='tight')
plt.show()


# <br>
# <br>

# ## Budowanie drzewa decyzyjnego

# In[38]:


from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree_model,
                      test_idx=range(105, 150))

plt.xlabel('Długość płatka [cm]')
plt.ylabel('Szerokość płatka [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('rysunki/03_20.png', dpi=300)
plt.show()


# In[39]:


from sklearn import tree

tree.plot_tree(tree_model)
#plt.savefig('rysunki/03_21_1.pdf')
plt.show()


# <br>
# <br>

# In[40]:


from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree_model,
                           filled=True, 
                           rounded=True,
                           class_names=['Setosa', 
                                        'Versicolor',
                                        'Virginica'],
                           feature_names=['Długość płatka', 
                                          'Szerokość płatka'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('drzewo.png') 


# In[41]:


Image(filename='rysunki/03_21.png', width=600) 


# <br>
# <br>

# ## Łączenie wielu drzew decyzyjnych za pomocą modelu losowego lasu

# In[42]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('Długość płatka [cm]')
plt.ylabel('Szerokość płatka [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('rysunki/03_22.png', dpi=300)
plt.show()


# <br>
# <br>

# # Algorytm k-najbliższych sąsiadów — model leniwego uczenia

# In[43]:


Image(filename='rysunki/03_23.png', width=400) 


# In[44]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('Długość płatka [standaryzowana]')
plt.ylabel('Szerokość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('rysunki/03_24.png', dpi=300)
plt.show()


# <br>
# <br>

# # Podsumowanie

# ...

# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.

# In[45]:


get_ipython().system(' python ../.convert_notebook_to_script.py --input r03.ipynb --output r03.py')


# In[ ]:




