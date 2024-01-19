Python. Uczenie maszynowe - kod źródłowy


##  Rozdział 10. Przewidywanie ciągłych zmiennych docelowych za pomocą analizy regresywnej

### Spis treści

- Wprowadzenie do regresji liniowej
  - Prosta regresja liniowa
  - Wielowymiarowa regresja liniowa
- Zestaw danych Housing
  - Wczytywanie zestawu danych Housing do obiektu DataFrame
  - Wizualizowanie ważnych elementów zestawu danych
  - Analiza związków za pomocą macierzy korelacji
- Implementacja modelu regresji liniowej wykorzystującego zwykłą metodę najmniejszych kwadratów
  - Określanie parametrów regresywnych za pomocą metody gradientu prostego
  - Szacowanie współczynnika modelu regresji za pomocą biblioteki scikit-learn
- Uczenie odpornego modelu regresywnego za pomocą algorytmu RANSAC
- Ocenianie skuteczności modeli regresji liniowej
- Stosowanie regularyzowanych metod regresji 
- Przekształcanie modelu regresji liniowej w krzywą — regresja wielomianowa
  - Dodawanie członów wielomianowych za pomocą biblioteki scikit-learn
  - Modelowanie nieliniowych zależności w zestawie danych Housing
- Analiza nieliniowych relacji za pomocą algorytmu losowego lasu
  - Regresja przy użyciu drzewa decyzyjnego
  - Regresja przy użyciu losowego lasu
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

**(nawet jeśli nie zamierzasz instalować aplikacji Jupyter Notebook, możesz przeglądać notatniki w serwisie GitHub. Wystarczy je kliknąć: [`r10.ipynb`](r10.ipynb))**

Oprócz samego kodu źródłowego, dołączyłem również w każdym notatniku Jupyter spis treści, a także nagłówki sekcji, które są spójne z treścią książki. Ponadto umieściłem również występujące w niej rysunki, dzięki czemu powinno Ci się łatwiej przeglądać zawartość plików i pracować z kodem.

![](../r02/rysunki/jupyter-przyklad-2.png)


Tworząc te notatniki przyświecał mi cel jak największego ułatwienia Tobie ich przeglądania (i tworzenia kodu)! Jeśli jednak nie zamierzasz korzystać z aplikacji Jupyter Notebook, przekonwertowałem te notatniki również do postaci standardowych plików skryptowych Pythona (w formacie `.py`), które można przeglądać i edytować w dowolnym edytorze tekstu.