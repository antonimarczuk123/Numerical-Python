Python. Uczenie maszynowe - kod źródłowy


##  Rozdział 13. Równoległe przetwarzanie sieci neuronowych za pomocą biblioteki TensorFlow


### Spis treści

- Biblioteka TensorFlow a skuteczność uczenia
  - Wyzwania związane z wydajnością
  - Czym jest biblioteka TensorFlow?
  - W jaki sposób będziemy poznawać bibliotekę TensorFlow?
- Pierwsze kroki z biblioteką TensorFlow
  - Instalacja modułu TensorFlow
  - Tworzenie tensorów w TensorFlow
  - Manipulowanie typem danych i rozmiarem tensora
  - Przeprowadzanie operacji matematycznych na tensorach
  - Dzielenie, nawarstwianie i łączenie tensorów
- Tworzenie potoków wejściowych za pomocą tf.data, czyli interfejsu danych TensorFlow
  - Tworzenie obiektów Dataset z istniejących tensorów
  - Łączenie dwóch tensorów we wspólny zestaw danych
  - Potasuj, pogrupuj, powtórz
  - Tworzenie zestawu danych z plików umieszczonych w lokalnym magazynie dyskowym
  - Pobieranie dostępnych zestawów danych z biblioteki tensorflow_datasets
- Tworzenie modelu sieci neuronowej za pomocą modułu TensorFlow
  - Interfejs Keras (tf.keras)
  - Tworzenie modelu regresji liniowej
  - Uczenie modelu za pomocą metod .compile() i .fit() 
  - Tworzenie perceptronu wielowarstwowego klasyfikującego kwiaty z zestawu danych Iris
  - Ocena wytrenowanego modelu za pomocą danych testowych
  - Zapisywanie i wczytywanie wyuczonego modelu
- Dobór funkcji aktywacji dla wielowarstwowych sieci neuronowych
  - Funkcja logistyczna — powtórzenie
  - Szacowanie prawdopodobieństw przynależności do klas w klasyfikacji wieloklasowej za pomocą funkcji softmax
  - Rozszerzanie zakresu wartości wyjściowych za pomocą funkcji tangensa hiperbolicznego
  - Aktywacja za pomocą prostowanej jednostki liniowej (ReLU)
- Podsumowanie

### Informacje na temat korzystania z kodu źródłowego

Zalecanym sposobem przeglądania kodu źródłowego opisywanego w książce jest aplikacja Jupyter Notebook (pliki w formacie `.ipynb`). W ten sposób jesteś w stanie realizować poszczególne fragmenty kodu krok po kroku, a wszystkie wyniki (łącznie z wykresami i rysunkami) są wygodnie generowane w jednym dokumencie.

![](../r02/rysunki/jupyter-przyklad-1.png)



Konfiguracja aplikacji Jupyter Notebook jest naprawdę prosta: jeżeli korzystasz z platformy Anaconda Python, wystarczy wpisać w terminalu poniższą komendę, aby zainstalować wspomniany program:

    conda install jupyter notebook

Teraz możesz uruchomić aplikację Jupyter Notebook w następujący sposób:

    jupyter notebook

Zostanie otwarte nowe okno w Twojej przeglądarce, w którym możesz przejść do katalogu docelowego zawierającego plik `.ipynb`, który zamierzasz otworzyć.

**Dodatkowe instrukcje dotyczące instalacji i konfiguracji znajdziesz w [pliku CZYTAJ.md w katalogu poświęconym rozdziałowi 1.](../r01/CZYTAJ.md)**.

**(nawet jeśli nie zamierzasz instalować aplikacji Jupyter Notebook, możesz przeglądać notatniki w serwisie GitHub. Wystarczy je kliknąć: [`r13_cz1.ipynb`](r13_cz1.ipynb) i [`r13_cz2.ipynb`](r13_cz2.ipynb))**

Oprócz samego kodu źródłowego, dołączyłem również w każdym notatniku Jupyter spis treści, a także nagłówki sekcji, które są spójne z treścią książki. Ponadto umieściłem również występujące w niej rysunki, dzięki czemu powinno Ci się łatwiej przeglądać zawartość plików i pracować z kodem.

![](../r02/rysunki/jupyter-przyklad-2.png)


Tworząc te notatniki przyświecał mi cel jak największego ułatwienia Tobie ich przeglądania (i tworzenia kodu)! Jeśli jednak nie zamierzasz korzystać z aplikacji Jupyter Notebook, przekonwertowałem te notatniki również do postaci standardowych plików skryptowych Pythona (w formacie `.py`), które można przeglądać i edytować w dowolnym edytorze tekstu. 
