# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from mlxtend.plotting import heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Python. Uczenie maszynowe - kod źródłowy

# # Rozdział 10. Przewidywanie ciągłych zmiennych docelowych za pomocą analizy regresywnej

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

# - [Wprowadzenie do regresji liniowej](#Wprowadzenie-do-regresji-liniowej)
#   - [Prosta regresja liniowa](#Prosta-regresja-liniowa)
#   - [Wielowymiarowa regresja liniowa](#Wielowymiarowa-regresja-liniowa)
# - [Zestaw danych Housing](#Zestaw-danych-Housing)
#   - [Wczytywanie zestawu danych Housing do obiektu DataFrame](#Wczytywanie-zestawu-danych-Housing-do-obiektu-DataFrame)
#   - [Wizualizowanie ważnych elementów zestawu danych](#Wizualizowanie-ważnych-elementów-zestawu-danych)
# - [Implementacja modelu regresji liniowej wykorzystującego zwykłą metodę najmniejszych kwadratów](#Implementacja-modelu-regresji-liniowej-wykorzystującego-zwykłą-metodę-najmniejszych-kwadratów)
#   - [Określanie parametrów regresywnych za pomocą metody gradientu prostego](#Określanie-parametrów-regresywnych-za-pomocą-metody-gradientu-prostego)
#   - [Szacowanie współczynnika modelu regresji za pomocą biblioteki scikit-learn](#Szacowanie-współczynnika-modelu-regresji-za-pomocą-biblioteki-scikit-learn)
# - [Uczenie odpornego modelu regresywnego za pomocą algorytmu RANSAC](#Uczenie-odpornego-modelu-regresywnego-za-pomocą-algorytmu-RANSAC)
# - [Ocenianie skuteczności modeli regresji liniowej](#Ocenianie-skuteczności-modeli-regresji-liniowej)
# - [Stosowanie regularyzowanych metod regresji](#Stosowanie-regularyzowanych-metod-regresji)
# - [Przekształcanie modelu regresji liniowej w krzywą — regresja wielomianowa](#Przekształcanie-modelu-regresji-liniowej-w-krzywą-—-regresja-wielomianowa)
#   - [Modelowanie nieliniowych zależności w zestawie danych Housing](#Modelowanie-nieliniowych-zależności-w-zestawie-danych-Housing)
#   - [Analiza nieliniowych relacji za pomocą algorytmu losowego lasu](#Analiza-nieliniowych-relacji-za-pomocą-algorytmu-losowego-lasu)
#     - [Regresja przy użyciu drzewa decyzyjnego](#Regresja-przy-użyciu-drzewa-decyzyjnego)
#     - [Regresja przy użyciu losowego lasu](#Regresja-przy-użyciu-losowego-lasu)
# - [Podsumowanie](#Podsumowanie)






# # Wprowadzenie do regresji liniowej

# ## Prosta regresja liniowa





# ## Wielowymiarowa regresja liniowa






# # Zestaw danych Housing

# ## Wczytywanie zestawu danych Housing do obiektu DataFrame

# Opis, dostępny uprzednio pod adresem [https://archive.ics.uci.edu/ml/datasets/Housing](https://archive.ics.uci.edu/ml/datasets/Housing)
# 
# Atrybuty:
#     
# <pre>
# 1. CRIM      per capita crime rate by town
# 2. ZN        proportion of residential land zoned for lots over 
#                  25,000 sq.ft.
# 3. INDUS     proportion of non-retail business acres per town
# 4. CHAS      Charles River dummy variable (= 1 if tract bounds 
#                  river; 0 otherwise)
# 5. NOX       nitric oxides concentration (parts per 10 million)
# 6. RM        average number of rooms per dwelling
# 7. AGE       proportion of owner-occupied units built prior to 1940
# 8. DIS       weighted distances to five Boston employment centres
# 9. RAD       index of accessibility to radial highways
# 10. TAX      full-value property-tax rate per $10,000
# 11. PTRATIO  pupil-teacher ratio by town
# 12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
#                  by town
# 13. LSTAT    % lower status of the population
# 14. MEDV     Median value of owner-occupied homes in $1000s
# </pre>




df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition/'
                 'master/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()


# 
# ### Uwaga
# 
# Uwaga
# Kopię zestawu danych Housing (a także wszystkich pozostałych ze-stawów danych wykorzystywanych w tej książce) znajdziesz w przy-kładowym kodzie dołączonym do niniejszej książki, dzięki czemu możesz z niego korzystać będąc odłączonym od internetu lub jeśli adres https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data będzie w danym momencie niedostępny. Na przykład, aby wczytać zestaw danych Housing z katalogu lokalnego, wystarczy zastąpić wiersze
# 
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#                  'machine-learning-databases'
#                  '/housing/housing.data',
#                  sep='\s+')
# 
# w powyższym przykładzie wierszem
# 
# df = pd.read_csv('./housing.data',
#                  sep='\s+')


# ## Wizualizowanie ważnych elementów zestawu danych







cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                  names=cols, alpha=0.5)
plt.tight_layout()
#plt.savefig('rysunki/10_03.png', dpi=300)
plt.show()






cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)

# plt.savefig('rysunki/10_04.png', dpi=300)
plt.show()



# # Implementacja modelu regresji liniowej wykorzystującego zwykłą metodę najmniejszych kwadratów

# ...

# ## Określanie parametrów regresywnych za pomocą metody gradientu prostego



class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)




X = df[['RM']].values
y = df['MEDV'].values






sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()




lr = LinearRegressionGD()
lr.fit(X_std, y_std)




plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('Suma kwadratów błędów')
plt.xlabel('Epoka')
#plt.tight_layout()
#plt.savefig('rysunki/10_05.png', dpi=300)
plt.show()




def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 




lin_regplot(X_std, y_std, lr)
plt.xlabel('Uśredniona liczba pomieszczeń [RM] (standaryzowana)')
plt.ylabel('Cena w tysiącach dolarów [MEDV] (standaryzowana)')

#plt.savefig('rysunki/10_06.png', dpi=300)
plt.show()




print('Nachylenie: %.3f' % lr.w_[1])
print('Punkt przecięcia: %.3f' % lr.w_[0])




num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print("Cena w tysiącach dolarów: %.3f" % sc_y.inverse_transform(price_std))



# ## Szacowanie współczynnika modelu regresji za pomocą biblioteki scikit-learn







slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Nachylenie: %.3f' % slr.coef_[0])
print('Punkt przecięcia: %.3f' % slr.intercept_)




lin_regplot(X, y, slr)
plt.xlabel('Uśredniona liczba pomieszczeń [RM]')
plt.ylabel('Cena w tysiącach dolarów [MEDV]')

#plt.savefig('rysunki/10_07.png', dpi=300)
plt.show()


# **Równania normalne** wersja alternatywna:



# Dodaje wektor kolumnowy zawierający "jedynki"
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print('Nachylenie: %.3f' % w[1])
print('Punkt przecięcia: %.3f' % w[0])



# # Uczenie odpornego modelu regresywnego za pomocą algorytmu RANSAC




ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=100, 
                         min_samples=50, 
                         loss='absolute_loss', 
                         residual_threshold=5.0, 
                         random_state=0)


ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white', 
            marker='o', label='Punkty nieodstające')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white', 
            marker='s', label='Punkty odstające')
plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Uśredniona liczba pomieszczeń [RM]')
plt.ylabel('Cena w tysiącach dolarów [MEDV]')
plt.legend(loc='upper left')

#plt.savefig('rysunki/10_08.png', dpi=300)
plt.show()




print('Nachylenie: %.3f' % ransac.estimator_.coef_[0])
print('Punkt przecięcia: %.3f' % ransac.estimator_.intercept_)



# # Ocenianie skuteczności modeli regresji liniowej




X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)




slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)





ary = np.array(range(100000))
















plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Dane uczące')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Dane testowe')
plt.xlabel('Przewidywane wartości')
plt.ylabel('Wartości resztowe')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('rysunki/10_09.png', dpi=300)
plt.show()





print('MSE na przykładach uczących: %.3f, testowych: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('Współczynnik R^2 dla danych uczących: %.3f, testowych: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))



# # Stosowanie regularyzowanych metod regresji




lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
print(lasso.coef_)




print('MSE na przykładach uczących: %.3f, testowych: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('Współczynnik R^2 dla danych uczących: %.3f, testowych: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# Regresja grzbietowa:



ridge = Ridge(alpha=1.0)


# Regresja typu LASSO:



lasso = Lasso(alpha=1.0)


# Regresja typu siatka elastyczna:



elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)



# # Przekształcanie modelu regresji liniowej w krzywą — regresja wielomianowa



X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])\
             [:, np.newaxis]

y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2,
              390.8])





lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)




# dopasowuje cechy liniowe
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# dopasowuje cechy kwadratowe
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# rysuje wykres wynikowy
plt.scatter(X, y, label='Punkty uczące')
plt.plot(X_fit, y_lin_fit, label='Dopasowanie liniowe', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='Dopasowanie kwadratowe')
plt.xlabel('Zmienna objaśniająca')
plt.ylabel('Przewidywane lub znane wartości docelowe')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('rysunki/10_11.png', dpi=300)
plt.show()




y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)




print('Błąd MSE dla uczenia liniowego: %.3f, kwadratowego: %.3f' % (
        mean_squared_error(y, y_lin_pred),
        mean_squared_error(y, y_quad_pred)))
print('Parametr R^2 dla uczenia liniowego: %.3f, kwadratowego: %.3f' % (
        r2_score(y, y_lin_pred),
        r2_score(y, y_quad_pred)))



# ## Modelowanie nieliniowych zależności w zestawie danych Housing



X = df[['LSTAT']].values
y = df['MEDV'].values

regr = LinearRegression()

# tworzy cechy kwadratowe
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# dopasowuje cechy
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))


# tworzy wykres wynikowy
plt.scatter(X, y, label='Punkty uczące', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='Liniowe (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', 
         lw=2, 
         linestyle=':')

plt.plot(X_fit, y_quad_fit, 
         label='Kwadratowe (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red', 
         lw=2,
         linestyle='-')

plt.plot(X_fit, y_cubic_fit, 
         label='Sześcienne (d=3), $R^2=%.2f$' % cubic_r2,
         color='green', 
         lw=2, 
         linestyle='--')

plt.xlabel('Odsetek uboższej części społeczeństwa  [LSTAT]')
plt.ylabel('Cena w tysiącach dolarów [MEDV]')
plt.legend(loc='upper right')

#plt.savefig('rysunki/10_12.png', dpi=300)
plt.show()


# Przekształcanie zestawu danych:



X = df[['LSTAT']].values
y = df['MEDV'].values

# przekształca cechy
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# dopasowuje cechy
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]

regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# tworzy wykres wynikowy
plt.scatter(X_log, y_sqrt, label='Punkty uczące', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='Liniowe (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', 
         lw=2)

plt.xlabel('log(odsetek uboższej części społeczeństwa  [LSTAT])')
plt.ylabel('$\sqrt{Cena \; w \; tysiącach \; dolarów \; [MEDV]}$')
plt.legend(loc='lower left')

plt.tight_layout()
#plt.savefig('rysunki/10_13.png', dpi=300)
plt.show()



# # Analiza nieliniowych relacji za pomocą algorytmu losowego lasu

# ...

# ## Regresja przy użyciu drzewa decyzyjnego




X = df[['LSTAT']].values
y = df['MEDV'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort()

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('Odsetek uboższej części społeczeństwa [LSTAT]')
plt.ylabel('Cena w tysiącach dolarów [MEDV]')
#plt.savefig('rysunki/10_14.png', dpi=300)
plt.show()



# ## Regresja przy użyciu losowego lasu



X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)





forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', 
                               random_state=1, 
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE na danych uczących: %.3f, testowych: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('Współczynnik R^2 dla danych uczących: %.3f, testowych: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))




plt.scatter(y_train_pred,  
            y_train_pred - y_train, 
            c='steelblue',
            edgecolor='white',
            marker='o', 
            s=35,
            alpha=0.9,
            label='Dane uczące')
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='limegreen',
            edgecolor='white',
            marker='s', 
            s=35,
            alpha=0.9,
            label='Dane testowe')

plt.xlabel('Przewidywane wartości')
plt.ylabel('Wartości resztowe')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.tight_layout()

#plt.savefig('rysunki/10_15.png', dpi=300)
plt.show()



# # Podsumowanie

# ...

# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.




