# coding: utf-8


import os
import sys
import tarfile
import time
import urllib.request
import pyprind
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import gzip
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import LatentDirichletAllocation

# Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)
# 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Python. Uczenie maszynowe - kod źródłowy

# # Chapter 8 - Applying Machine Learning To Sentiment Analysis

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

# - [Przygotowywanie zestawu danych IMDb movie review do przetwarzania tekstu](#Przygotowywanie-zestawu-danych-IMDb-movie-review-do-przetwarzania-tekstu)
#   - [Uzyskiwanie zestawu danych IMDb](#Uzyskiwanie-zestawu-danych-IMDb)
#   - [Przetwarzanie wstępne zestawu danych IMDb do wygodniejszego formatu](#Przetwarzanie-wstępne-zestawu-danych-IMDb-do-wygodniejszego-formatu)
# - [Wprowadzenie do modelu worka słów](#Wprowadzenie-do-modelu-worka-słów)
#   - [Przekształcanie słów w wektory cech](#Przekształcanie-słów-w-wektory-cech)
#   - [Ocena istotności wyrazów za pomocą ważenia częstości termów — odwrotnej częstości w tekście](#Ocena-istotności-wyrazów-za-pomocą-ważenia-częstości-termów-—-odwrotnej-częstości-w-tekście)
#   - [Oczyszczanie danych tekstowych](#Oczyszczanie-danych-tekstowych)
#   - [Przetwarzanie tekstu na znaczniki](#Przetwarzanie-tekstu-na-znaczniki)
# - [Uczenie modelu regresji logistycznej w celu klasyfikowania tekstu](#Uczenie-modelu-regresji-logistycznej-w-celu-klasyfikowania-tekstu)
# - [Praca z większą ilością danych — algorytmy sieciowe i uczenie pozardzeniowe](#Praca-z-większą-ilością-danych-—-algorytmy-sieciowe-i-uczenie-pozardzeniowe)
# - [Modelowanie tematyczne za pomocą alokacji ukrytej zmiennej Dirichleta](#Modelowanie-tematyczne-za-pomocą-alokacji-ukrytej-zmiennej-Dirichleta)
#   - [Rozkładanie dokumentów tekstowych za pomocą analizy LDA](#Rozkładanie-dokumentów-tekstowych-za-pomocą-analizy-LDA)
#   - [Analiza LDA w bibliotece scikit-learn](#Analiza-LDA-w-bibliotece-scikit-learn)
# - [Podsumowanie](#Podsumowanie)


# # Przygotowywanie zestawu danych IMDb movie review do przetwarzania tekstu

# ## Uzyskiwanie zestawu danych IMDb

# Zestaw danych IMDB movie review jest dostępny do pobrania pod adresem [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/).
# Po pobraniu pliku należy go rozpakować.
# 
# A) Jeżeli używasz systemu Linux lub MacOS X, otwórz nowe okno terminala i za pomocą polecenia `cd` przejdź do katalogu, w którym znajduje się pobrane archiwum, a następnie wpisz komendę: 
# 
# `tar -zxf aclImdb_v1.tar.gz`
# 
# B) W przypadku systemu Windows pobierz archiwizator (np. [7Zip](http://www.7-zip.org)), dzięki któremu otworzysz skompresowany plik.

# **Alternatywny kod służący do pobrania i rozpakowania zestawu danych w środowisku Python:**





source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
target = 'aclImdb_v1.tar.gz'


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.**2 * duration)
    percent = count * block_size * 100. / total_size

    sys.stdout.write("\r%d%% | %d MB | %.2f MB/s | %d s szacowanego czasu" %
                    (percent, progress_size / (1024.**2), speed, duration))
    sys.stdout.flush()


if not os.path.isdir('aclImdb') and not os.path.isfile('aclImdb_v1.tar.gz'):
    urllib.request.urlretrieve(source, target, reporthook)




if not os.path.isdir('aclImdb'):

    with tarfile.open(target, 'r:gz') as tar:
        tar.extractall()


# ## Przetwarzanie wstępne zestawu danych IMDb do wygodniejszego formatu




# zmień wartość zmiennej `basepath` na ścieżkę do katalogu, w którym
# znajduje się nierozpakowany plik z zestawem danych

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], 
                           ignore_index=True)
            pbar.update()
df.columns = ['Recenzja', 'Sentyment']


# Tasowanie zawartości obiektu DataFrame:




np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))


# Opcjonalnie: Zapisywanie zgromadzonych danych w postaci pliku CSV:



df.to_csv('movie_data.csv', index=False, encoding='utf-8')





df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)




df.shape


# ---
# 
# ### Uwaga
# 
# Jeżeli masz problem z utworzeniem pliku `movie_data.csv`, możesz pobrać jego zarchiwizowaną postać ze strony 
# https://github.com/rasbt/python-machine-learning-book-3rd-edition/tree/master/code/ch08/.
# 
# ---


# # Wprowadzenie do modelu worka słów

# ...

# ## Przekształcanie słów w wektory cech

# Wywołując metodę fit_transform w CountVectorizer stworzyliśmy wokabularz modelu worka słów i przekształciliśmy trzy następujące zdanie w rzadkie wektory cech:
# 
# 1. Słońce grzeje dziś mocno
# 2. Pogoda jest dziś wiosenna
# 3. Słońce grzeje dziś mocno i pogoda jest dziś wiosenna, a jeden i jeden daje dwa
# 




count = CountVectorizer()
docs = np.array([
        'Słońce grzeje dziś mocno',
        'Pogoda jest dziś wiosenna',
        'Słońce grzeje dziś mocno i pogoda jest dziś wiosenna, a jeden i jeden daje dwa'])
bag = count.fit_transform(docs)


# Sprawdźmy teraz zawartość tego wokabularza, aby lepiej zrozumieć pojęcia, jakimi będziemy się posługiwać:



print(count.vocabulary_)


# Jak widać powyżej, wokabularz jest przechowywany w słowniku Pythona, w którym poszczególne wyrazy otrzymują indeksy liczbowe. Spójrzmy na wygenerowane przez nas wektory cech:

# Pozycja każdego indeksu w ukazanych wektorach cech odpowiada liczbom całkowitym przechowywanym jako elementy słownika CountVectorizer; np. pierwsza cecha o indeksie 0 reprezentuje zliczenia wyrazu „daje”, który pojawia się wyłącznie w ostatnim zdaniu, natomiast słowo „dziś” na trzeciej pozycji (indeks 2 w wektorach cech) występuje we wszystkich trzech zdaniach. Wartości przechowywane w wektorach cech są również nazywane częstością termów: *tf (t, d)* — liczbą wystąpień termu t w dokumencie *d*.



print(bag.toarray())



# ## Ocena istotności wyrazów za pomocą ważenia częstości termów — odwrotnej częstości w tekście



np.set_printoptions(precision=2)


# W czasie analizowania danych tekstowych często natrafiamy na wyrazy z obydwu klas pojawiające się w wielu różnych danych tekstowych. Takie często występujące słowa zazwyczaj nie zawierają żadnych przydatnych ani rozróżniających informacji. W tym ustępie poznamy użyteczną technikę zwaną ważeniem częstości termów — odwrotną częstością w tekście (tf-idf), za pomocą której zmniejszamy wagi takich mniej istotnych wyrazów w wektorach cech. Parametr tf-idf możemy zdefiniować jako iloczyn częstości termów przez odwrotną częstość w tekście:
# 
# $$\text{tf-idf}(t,d)=\text{tf (t,d)}\times \text{idf}(t,d)$$
# 
# Tutaj tf (t, d) jest wprowadzoną w poprzednim ustępie częstością termów, natomiast odwrotną częstość w tekście idf (t, d) defi-niujemy następująco:
# 
# $$\text{idf}(t,d) = \text{log}\frac{n_d}{1+\text{df}(d, t)},$$
# 
# gdzie $n_d$ oznacza całkowitą liczbę danych tekstowych, a *df(d, t)* — liczbę danych tekstowych *d* zawierających term *t*. Zwróć uwagę, że wstawienie stałej 1 do mianownika nie jest konieczne i służy do przydzielania niezerowej wartości termom znajdującym się we wszystkich próbkach uczących; logarytm gwarantuje, że niewielkie częstości danych tekstowych nie będą otrzymywały zbyt dużej wagi.
# 
# Biblioteka scikit-learn zawiera jeszcze jedną klasę transformującą, `TfidfTransformer`, przyjmującą częstości termów przechowywane w klasie `CountVectorizer` i przekształcające je w wartości tf-idfs:




tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', 
                         smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())


# Jak wiemy z poprzedniego ustępu, wyraz „dziś” ma największą częstość termu w trzecim dokumencie, przez co stanowi najczęściej występujące słowo. Jednak po przekształceniu tego samego wektora cech za pomocą algorytmu tf-idf widzimy, że słowo to jest powiązane teraz ze względnie niską wartością tf-idf (0,36) w trzecim dokumencie, ponieważ występuje również w pozostałych dwóch dokumentach tekstowych, zatem jest mało prawdopodobne, że jest ono nośnikiem przydatnych, rozróżniających informacji.
# 

# Gdybyśmy jednak ręcznie policzyli wartości tf-idf poszczególnych termów w wektorach cech, okazałoby się, że klasa `TfidfTransformer` oblicza je nieco inaczej niż standardowe, omówione wcześniej wzory. Równanie na odwrotną częstość tekstów zaimplementowane w bibliotece scikit-learn wygląda następująco:

# $$\text{idf} (t,d) = log\frac{1 + n_d}{1 + \text{df}(d, t)}$$
# 
# Z kolei wzór na model tf-idf używany w bibliotece scikit-learn wygląda następująco:
# 
# $$\text{tf-idf}(t,d) = \text{tf}(t,d) \times (\text{idf}(t,d)+1)$$
# 
# Zazwyczaj częstości termów są normalizowane jeszcze przed obliczeniem wartości tf-idf, ale klasa `TfidfTransformer` normalizuje wyniki tf-idf bezpośrednio.
# 
# Domyślnie (`norm='l2'`) przeprowadzana jest normalizacja L2, która zwraca wektor o długości 1, poprzez podzielenie nieznormalizowanego wektora cech v przez normę L2:
# 
# $$v_{\text{norm}} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v_{1}^{2} + v_{2}^{2} + \dots + v_{n}^{2}}} = \frac{v}{\big (\sum_{i=1}^{n} v_{i}^{2}\big)^\frac{1}{2}}$$
# 
# Aby upewnić się, że rozumiemy mechanizm działania klasy `TfidfTransformer`, przyjrzyjmy się przykładowi i wyliczmy wartość tf-idf wyrazu „dziś” w trzecim dokumencie.
# 
# Wyraz „dziś” ma w trzecim dokumencie częstość termu równą 2 (tf = 2), jego częstość występowania w tekście również wynosi 3, ponieważ znajduje się we wszystkich trzech zdaniach (df = 3). Możemy więc wyliczyć parametr idf następująco:
# 
# $$\text{idf}("dziś", d3) = log \frac{1+3}{1+3} = 0$$
# 
# Teraz w celu obliczenia wartości tf-idf wystarczy dodać 1 do odwrotnej częstości w dokumencie i pomnożyć wynik przez częstość termu:
# 
# $$\text{tf-idf}("dziś",d3)= 2 \times (0+1) = 2$$



tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('wartość tf-idf termu "dziś" = %.2f' % tfidf_is)


# Po przeprowadzeniu analogicznych obliczeń dla pozostałych termów w trzecim dokumencie uzyskamy następujący wektor tf-idf: [
# 1,69, 1,69, 2  , 1,29, 3,39, 1,29, 1,29, 1,29, 1,29, 1,29]. Widać jednak, że wartości w tym wektorze cech różnią się od wyników uzyskanych za pomocą klasy `TfidfTransformer`. Musimy jeszcze tylko przeprowadzić normalizację L2, którą wykonamy w sposób zaprezentowany poniżej:

# $$\text{tfi-df}_{norm} = \frac{[1.69, 1.69, 2.0, 1.29, 3.39, 1.29, 1.29 , 1.29, 1.29, 1.29]}{\sqrt{[1.69^2, 1.69^2, 2.0^2, 1.29^2, 3.39^2, 1.29^2, 1.29^2 , 1.29^2, 1.29^2, 1.29^2]}}$$
# 
# $$=[0.3, 0.3, 0.36, 0.23, 0.61, 0.23, 0.23, 0.23, 0.23, 0.23]$$
# 
# $$\Rightarrow \text{tfi-df}_{norm}("dziś", d3) = 0.45$$

# Jak widać, uzyskane wyniki są zgodne z rezultatami zwracanymi przez klasę `TfidfTransformer` (poniżej). Skoro już wiemy, jak są wyliczane wartości tf-idf, możemy przejść do dalszej części rozdziału i wykorzystać omówione koncepcje w odniesieniu do naszego zestawu recenzji filmów.



tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf 




l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf



# ## Oczyszczanie danych tekstowych



df.loc[0, 'Recenzja'][-50:]




def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text




preprocessor(df.loc[0, 'Recenzja'][-50:])




preprocessor("</a>To :) jest :( test :-)!")




df['Recenzja'] = df['Recenzja'].apply(preprocessor)



# ## Przetwarzanie tekstu na znaczniki




porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]




tokenizer('runners like running and thus they run')




tokenizer_porter('runners like running and thus they run')





nltk.download('stopwords')





stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]



# # Uczenie modelu regresji logistycznej w celu klasyfikowania tekstu

# Usuwa znaczniki HTML i znaki interpunkcyjne w celu przyspieszenia przeszukiwania siatki:



X_train = df.loc[:25000, 'Recenzja'].values
y_train = df.loc[:25000, 'Sentyment'].values
X_test = df.loc[25000:, 'Recenzja'].values
y_test = df.loc[25000:, 'Sentyment'].values





tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)


# **Ważna uwaga na temat parametru `n_jobs`**
# 
# Bardzo zalecane jest użycie parametru `n_jobs=-1` (zamiast `n_jobs=1`) w powyższym listingu aby wykorzystać wszystkie dostępne rdzenie procesora i przyśpieszyć przeszukiwanie siatki. Jednak część użytkowników systemu Windows zgłaszała problemy z działaniem kodu przy wyznaczonym parametrze `n_jobs=-1` co ma związek z wielordzeniowym konserwowaniem funkcji `tokenizer` i `tokenizer_porter` w systemie Windows. Można ten problem obejść zastępując funkcje  `[tokenizer, tokenizer_porter]` funkcją `[str.split]`. W konsekwencji jednak nie będziemy mogli korzystać z rdzeniowania wyrazów.

# **Ważna uwaga na temat czasu przetwarzania kodu**
# 
# Realizacja poniższej komórki **może zająć od 30 do 60 minut** w zależności od konfiguracji sprzętowej, ponieważ zgodnie ze zdefiniowanymi parametrami siatki istnieje 2*2*2*3*5 + 2*2*2*3*5 = 240 modeli do wytrenowania.
# 
# Jeżeli nie chcesz czekać tak długo, możesz zmniejszyć rozmiar zestawu danych ograniczając liczbę przykładów uczących, np. w następujący sposób:
# 
#     X_train = df.loc[:2500, 'Recenzja'].values
#     y_train = df.loc[:2500, 'Sentyment'].values
#     
# Pamiętaj jednak, że tak ograniczony rozmiar zestawu uczącego będzie oznaczał kiepską skuteczność modeli. Możesz ewentualnie usunąć parametry z powyższej siatki, aby zmniejszyć liczbę modeli - np. tak, jak pokazano poniżej:
# 
#     param_grid = [{'vect__ngram_range': [(1, 1)],
#                    'vect__stop_words': [stop, None],
#                    'vect__tokenizer': [tokenizer],
#                    'clf__penalty': ['l1', 'l2'],
#                    'clf__C': [1.0, 10.0]},
#                   ]



gs_lr_tfidf.fit(X_train, y_train)




print('Zestaw najlepszych parametrów: %s ' % gs_lr_tfidf.best_params_)
print('Dokładność sprawdzianu krzyżowego: %.3f' % gs_lr_tfidf.best_score_)




clf = gs_lr_tfidf.best_estimator_
print('Dokładność testu: %.3f' % clf.score(X_test, y_test))



# ####  Początek komentarza:
#     
# Zwróć uwagę, że `gs_lr_tfidf.best_score_` daje uśredniony wynik k-krotnego sprawdzianu krzyżowego. Np. gdybyśmy korzystali z obiektu `GridSearchCV` składającego się z pięciokrotnej kroswalidacji (tak jak w powyższym przykładzie), atrybut `best_score_` będzie zwracał uśredniony wynik po wyznaczeniu najlepszego modelu za pomocą pięciokrotnej walidacji krzyżowej. Wyjaśnię to na przykładzie:





np.random.seed(0)
np.set_printoptions(precision=6)
y = [np.random.randint(3) for i in range(25)]
X = (y + np.random.randn(25)).reshape(-1, 1)

cv5_idx = list(StratifiedKFold(n_splits=5, shuffle=False).split(X, y))
    
lr = LogisticRegression(random_state=123, multi_class='ovr', solver='lbfgs')
cross_val_score(lr, X, y, cv=cv5_idx)


# Po uruchomieniu powyższego kodu stworzyliśmy prosty zestaw danych składający się z losowych liczb całkowitych, które będą symbolizować etykiety klas. Następnie przekazaliśmy indeksy podzbiorów pięciokrotnego sprawdzianu krzyżowego (`cv3_idx`) funkcji zliczającej `cross_val_score`, która zwróciła pięć wyników dokładności - jest to pięć wartości wyliczonych dla poszczególnych podzbiorów testowych.
# 
# Użyjmy teraz obiektu `GridSearchCV` i przekażmy mu te same pięć podzbiorów pięciokrotnej kroswalidacji (poprzez indeksy `cv3_idx`):




lr = LogisticRegression(solver='lbfgs', multi_class='ovr', random_state=1)
gs = GridSearchCV(lr, {}, cv=cv5_idx, verbose=3).fit(X, y) 


# Jak widać, wyniki dla pięciu podzbiorów są identyczne, jak wyliczone wcześniej przez funkcję `cross_val_score`.

# Atrybut `best_score_` (dostępny po wytrenowaniu klasyfikatora) obiektu `GridSearchCV` zwraca uśredniony wynik dokładności dla najlepszego modelu:



gs.best_score_


# Wynik ten jest, jak widać, zgodny z uśrednionym wynikiem dokładności wyliczonym za pomocą funkcji `cross_val_score`.



lr = LogisticRegression(solver='lbfgs', multi_class='ovr', random_state=1)
cross_val_score(lr, X, y, cv=cv5_idx).mean()


# #### Koniec komentarza.
# 


# # Praca z większą ilością danych — algorytmy sieciowe i uczenie pozardzeniowe



# Komórka ta nie jest opisana w książce, lecz
# została dodana dla Twojej wygody, żebyć mógł
# ropocząć realizowanie kodu od tego miejsca, bez
# konieczności uruchamiania wcześniejszych komórek.



if not os.path.isfile('movie_data.csv'):
    if not os.path.isfile('movie_data.csv.gz'):
        print('Umieść tu kopię archiwum movie_data.csv.gz'
              'Uzyskasz do niej dostęp'
              'a) realizując kod na początku tego notatnika'
              'lub b) pobierając go z serwisu GitHub:'
              'https://github.com/rasbt/python-machine-learning-'
              'book-2nd-edition/blob/master/code/ch08/movie_data.csv.gz')
    else:
        with gzip.open('movie_data.csv.gz', 'rb') as in_f,                 open('movie_data.csv', 'wb') as out_f:
            out_f.write(in_f.read())






# Funkcja `stop` została zdefiniowana wcześniej w tym rozdziale.
# Umieściliśmy ją tu dla Twojej wygody, dzięki czemu niniejszy podrozdział
# może być traktowany niezależnie od pozostałych fragmentów notatnika w
# tym katalogu
stop = stopwords.words('english')


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # pomija nagłówek
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label




next(stream_docs(path='movie_data.csv'))




def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y






vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)





clf = SGDClassifier(loss='log', random_state=1)


doc_stream = stream_docs(path='movie_data.csv')




pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()




X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Dokładność: %.3f' % clf.score(X_test, y_test))




clf = clf.partial_fit(X_test, y_test)


# ## Modelowanie tematyczne za pomocą alokacji ukrytej zmiennej Dirichleta

# ### Rozkładanie dokumentów tekstowych za pomocą analizy LDA

# ### Analiza LDA w bibliotece scikit-learn




df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)





count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['Recenzja'].values)





lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)




lda.components_.shape




n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("Temat %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))


# Na podstawie pięciu najważniejszych wyrazów dla każdego tematu możemy zgadywać, że model LDA wykrył następujące tematy:
#     
# 1. Generalnie kiepskie filmy (w rzeczywistości nie jest to kategoria).
# 2. Filmy o rodzinie.
# 3.	Filmy wojenne.
# 4.	Filmy artystyczne.
# 5.	Kryminały.
# 6.	Horrory.
# 7.	Komedie.
# 8.	Seriale lub filmy powiązane z serialami.
# 9.	Adaptacje książek.
# 10.	Filmy sensacyjne.

# Aby upewnić się, że kategorie tworzone na podstawie recenzji mają sens, sprawdźmy trzy filmy oznaczone jako horrory (kategoria szósta, indeks 5):



horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror numer #%d:' % (iter_idx + 1))
    print(df['Recenzja'][movie_idx][:300], '...')


# Za pomocą powyższego kodu wyświetliliśmy pierwszych 300 znaków z recenzji trzech pierw-szych horrorów i widzimy, że recenzje te — mimo że nie znamy tytułów filmów — wyglądają na opisy horrorów (chociaż w przypadku horroru numer 2 równie dobrze moglibyśmy zaklasy-fikować go kategorii 1.: zasadniczo kiepskie filmy).


# # Podsumowanie

# ...

# ---
# 
# Czytelnicy mogą zignorować poniższą komórkę.




