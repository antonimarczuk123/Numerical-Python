# coding: utf-8


import pandas as pd
from io import StringIO
import sys
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Python. Uczenie maszynowe - kod źródłowy

# # Rozdział 4. Tworzenie dobrych zestawów danych uczących — wstępne przetwarzanie danych

# Zwróć uwagę, że rozszerzenie zawierające nieobowiązkowy znak wodny stanowi niewielki plugin notatnika IPython / Jupyter, który zaprojektowałem w celu powielania kodu źródłowego. Wystarczy pominąć poniższe wiersze kodu:





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

# - [Kwestia brakujących danych](#Kwestia-brakujących-danych)
#   - [Wykrywanie brakujących wartości w danych tabelarycznych](#Wykrywanie-brakujących-wartości-w-danych-tabelarycznych)
#   - [Usuwanie przykładów uczących lub cech niezawierających wartości](#Usuwanie-przykładów-uczących-lub-cech-niezawierających-wartości)
#   - [Wstawianie brakujących danych](#Wstawianie-brakujących-danych)
#   - [Estymatory interfejsu scikit-learn](#Estymatory-interfejsu-scikit-learn)
# - [Przetwarzanie danych kategorialnych](#Przetwarzanie-danych-kategorialnych)
#   - [Cechy nominalne i porządkowe](#Cechy-nominalne-i-porządkowe)
#   - [Mapowanie cech porządkowych](#Mapowanie-cech-porządkowych)
#   - [Kodowanie etykiet klas](#Kodowanie-etykiet-klas)
#   - [Kodowanie „gorącojedynkowe” cech nominalnych](#Kodowanie-„gorącojedynkowe”-cech-nominalnych)
# - [Rozdzielanie zestawu danych na oddzielne podzbiory uczące i testowe](#Rozdzielanie-zestawu-danych-na-oddzielne-podzbiory-uczące-i-testowe)
# - [Skalowanie cech](#Skalowanie-cech)
# - [Dobór odpowiednich cech](#Dobór-odpowiednich-cech)
#   - [Regularyzacje L1 i L2 jako kary ograniczające złożoność modelu](#Regularyzacje-L1-i-L2-jako-kary-ograniczające-złożoność-modelu)
#   - [Interpretacja geometryczna regularyzacji L2](#Interpretacja-geometryczna-regularyzacji-L2)
#   - [Rozwiązania rzadkie za pomocą regularyzacji L1](#Rozwiązania-rzadkie-za-pomocą-regularyzacji-L1)
#   - [Algorytmy sekwencyjnego wyboru cech](#Algorytmy-sekwencyjnego-wyboru-cech)
# - [Ocenianie istotności cech za pomocą algorytmu losowego lasu](#Ocenianie-istotności-cech-za-pomocą-algorytmu-losowego-lasu)
# - [Podsumowanie](#Podsumowanie)






# # Kwestia brakujących danych

# ## Wykrywanie brakujących wartości w danych tabelarycznych




csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# jeżeli korzystasz ze środowiska Python 2.7, musisz
# przekonwertować ciąg znaków do standardu unicode:

if (sys.version_info < (3, 0)):
    csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))
df




df.isnull().sum()




# uzyskujemy dostęp do tablicy NumPy
# za pomocą atrybutu `values`
df.values



# ## Usuwanie przykładów uczących lub cech niezawierających wartości



# usuwa wiersze, w których brakuje wartości

df.dropna(axis=0)




# usuwa kolumny, w których brakuje wartości

df.dropna(axis=1)




# usuwa jedynie wiersze, w których wszystkie kolumny mają wartość NaN

df.dropna(how='all')  




# usuwa wiersze, w których znajduje się mniej, niż trzy wartości rzeczywiste 

df.dropna(thresh=4)




# usuwa jedynie wiersze, dla których wartość NaN pojawia się w określonych kolumnach (tutaj w kolumnie 'C')

df.dropna(subset=['C'])



# ## Wstawianie brakujących danych



# nasza pierwotna tablica
df.values




# wstawia brakujące wartości wykorzystując średnią z kolumny


imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data





df.fillna(df.mean())


# ## Estymatory interfejsu scikit-learn










# # Przetwarzanie danych kategorialnych

# ## Cechy nominalne i porządkowe




df = pd.DataFrame([['Zielony', 'M', 10.1, 'klasa2'],
                   ['Czerwony', 'L', 13.5, 'klasa1'],
                   ['Niebieski', 'XL', 15.3, 'klasa2']])

df.columns = ['Kolor', 'Rozmiar', 'Cena', 'Etykieta klas']
df



# ## Mapowanie cech porządkowych



size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['Rozmiar'] = df['Rozmiar'].map(size_mapping)
df




inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['Rozmiar'].map(inv_size_mapping)



# ## Kodowanie etykiet klas




# tworzy słownik mapowania
# przekształcający etykiety klas z ciągów znaków do postaci liczb całkowitych
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['Etykieta klas']))}
class_mapping




# przekształca etykiety klas z ciągów znaków do postaci liczb całkowitych
df['Etykieta klas'] = df['Etykieta klas'].map(class_mapping)
df




# odwraca proces mapowania
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['Etykieta klas'] = df['Etykieta klas'].map(inv_class_mapping)
df





# kodowanie etykiet za pomocą klasy LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['Etykieta klas'].values)
y




# odwrotne mapowanie
class_le.inverse_transform(y)



# ## Kodowanie „gorącojedynkowe” cech nominalnych



X = df[['Kolor', 'Rozmiar', 'Cena']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X





X = df[['Kolor', 'Rozmiar', 'Cena']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()





X = df[['Kolor', 'Rozmiar', 'Cena']].values
c_transf = ColumnTransformer([ ('onehot', OneHotEncoder(), [0]),
                               ('nothing', 'passthrough', [1, 2])])
c_transf.fit_transform(X).astype(float)




# kodowanie gorącojedynkowe za pomocą biblioteki pandas

pd.get_dummies(df[['Kolor', 'Rozmiar', 'Cena']])




# ochrona współliniowości w funkcji get_dummies

pd.get_dummies(df[['Kolor', 'Rozmiar', 'Cena']], drop_first=True)




# ochrona współiniowości dla klasy OneHotEncoder

color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([ ('onehot', color_ohe, [0]),
                               ('nothing', 'passthrough', [1, 2])])
c_transf.fit_transform(X).astype(float)



# ## Bonus: kodowanie cech porządkowych

# Jeżeli nie jesteśmy pewni różnic numerycznych pomiędzy kategoriami cech porządkowych lub jeśli nie została zdefiniowana różnica pomiędzy dwiema cechami porządkowymi, możemy je kodować za pomocą kodowania progowego przy użyciu wartości 0/1. Na przykład możemy rozdzielić cechę "Rozmiar" mającą wartości M, L i XL na dwie nowe cechy: "x > M" i "x > L". Zacznijmy od naszej pierwotnej ramki danych:



df = pd.DataFrame([['Zielony', 'M', 10.1, 'klasa2'],
                   ['Czerwony', 'L', 13.5, 'klasa1'],
                   ['Niebieski', 'XL', 15.3, 'klasa2']])

df.columns = ['Kolor', 'Rozmiar', 'Cena', 'Etykieta klas']
df


# Możemy użyć metody `apply` do tworzenia niestandardowych wyrażeń lambda po to, aby kodować te zmienne za pomocą wartości progowych:



df['x > M'] = df['Rozmiar'].apply(lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['Rozmiar'].apply(lambda x: 1 if x == 'XL' else 0)

del df['Rozmiar']
df



# # Rozdzielanie zestawu danych na oddzielne podzbiory uczące i testowe



df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

# jeżeli zestaw danych Wine będzie tymczasowo niedostępny w repozytorium UCI,
# usuń znak komentarza z poniższego wiersza, aby wczytać ten zestaw z katalogu lokalnego:

# df_wine = pd.read_csv('wine.data', header=None)


df_wine.columns = ['Etykieta klas', 'Alkohol', 'Kwas jabłkowy', 'Popiół',
                   'Zasadowość popiołu', 'Magnez', 'Całk. zaw. fenoli',
                   'Flawonoidy', 'Fenole nieflawonoidowe', 'Proantocyjaniny',
                   'Intensywność koloru', 'Odcień', 'Transmitancja 280/315 nm',
                   'Prolina']

print('Etykiety klas', np.unique(df_wine['Etykieta klas']))
df_wine.head()





X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)



# # Skalowanie cech




mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)





stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# Przykład wizualny:



ex = np.array([0, 1, 2, 3, 4, 5])

print('Standaryzowane:', (ex - ex.mean()) / ex.std())

# Zwróć uwagę, że biblioteka pandas domyślnie wykorzystuje ddof=1 (odchylenie standardowe przykładu), 
# natomiast metoda std i klasa StandardScaler biblioteki NumPy
# używają ddof=0 (odchylenie standardowe populacji)

# normalizuje
print('Normalizowane:', (ex - ex.min()) / (ex.max() - ex.min()))



# # Dobór odpowiednich cech

# ...

# ## Regularyzacje L1 i L2 jako kary ograniczające złożoność modelu

# ## Interpretacja geometryczna regularyzacji L2









# ## Rozwiązania rzadkie za pomocą regularyzacji L1





# W przypadku regularyzowanych modeli wykorzystujących regularyzację L1, możemy wyznaczyć wartość `'l1'` parametru `penalty`, aby uzyskać rozwiązanie rzadkie:



LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr')


# Stosujemy wobec standaryzowanych danych Wine ...




lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
# Wartość C=1.0 jest domyślna. Możesz ją zmniejszać
# lub zwiększać aby, odpowiednio, osłabiać lub wzmacniać
# siłę regularyzacji.
lr.fit(X_train_std, y_train)
print('Dokładność dla danych uczących:', lr.score(X_train_std, y_train))
print('Dokładność dla danych testowych:', lr.score(X_test_std, y_test))




lr.intercept_




np.set_printoptions(8)




lr.coef_[lr.coef_!=0].shape




lr.coef_





fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', 
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Współczynnik wag')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
#plt.savefig('rysunki/04_07.png', dpi=300, 
#            bbox_inches='tight', pad_inches=0.2)
plt.show()



# ## Algorytmy sekwencyjnego wyboru cech





class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test =             train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score





knn = KNeighborsClassifier(n_neighbors=5)

# dobiera cechy
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# rysuje wykres wydajności podzbiorów cech
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Dokładność')
plt.xlabel('Liczba cech')
plt.grid()
plt.tight_layout()
# plt.savefig('rysunki/04_08.png', dpi=300)
plt.show()




k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])




knn.fit(X_train_std, y_train)
print('Dokładność dla danych uczących:', knn.score(X_train_std, y_train))
print('Dokładność dla danych testowych:', knn.score(X_test_std, y_test))




knn.fit(X_train_std[:, k3], y_train)
print('Dokładność dla danych uczących:', knn.score(X_train_std[:, k3], y_train))
print('Dokładność dla danych testowych:', knn.score(X_test_std[:, k3], y_test))



# # Ocenianie istotności cech za pomocą algorytmu losowego lasu




feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Istotność cech')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('rysunki/04_09.png', dpi=300)
plt.show()





sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Liczba cech spełniających dane kryterium progowe:', 
      X_selected.shape[1])


# Wyświetlmy teraz trzy cechy, które spełniły sformułowany powyżej warunek progowy doboru cech (poniższy fragment kodu nie jest obecny w książce, lecz został dodany do niniejszego notatnika w celach poglądowych):



for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))


# # Podsumowanie

# ...

# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.




