Python. Uczenie maszynowe - kod źródłowy


##  Rozdział 12. Implementowanie wielowarstwowej sieci neuronowej od podstaw

### Spis treści

- Modelowanie złożonych funkcji przy użyciu sztucznych sieci neuronowych
  - Jednowarstwowa sieć neuronowa — powtórzenie
  - Wstęp do wielowarstwowej architektury sieci neuronowych
  - Aktywacja sieci neuronowej za pomocą propagacji w przód
- Klasyfikowanie pisma odręcznego
  - Zestaw danych MNIST
  - Implementacja perceptronu wielowarstwowego
- Trenowanie sztucznej sieci neuronowej
  - Obliczanie logistycznej funkcji kosztu
  - Ujęcie intuicyjne algorytmu wstecznej propagacji 
  - Uczenie sieci neuronowych za pomocą algorytmu propagacji wstecznej
- Zbieżność w sieciach neuronowych
- Jeszcze słowo o implementacji sieci neuronowej
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

**(nawet jeśli nie zamierzasz instalować aplikacji Jupyter Notebook, możesz przeglądać notatniki w serwisie GitHub. Wystarczy je kliknąć: [`r12.ipynb`](r12.ipynb))**

Oprócz samego kodu źródłowego, dołączyłem również w każdym notatniku Jupyter spis treści, a także nagłówki sekcji, które są spójne z treścią książki. Ponadto umieściłem również występujące w niej rysunki, dzięki czemu powinno Ci się łatwiej przeglądać zawartość plików i pracować z kodem.

![](../r02/rysunki/jupyter-przyklad-2.png)


Tworząc te notatniki przyświecał mi cel jak największego ułatwienia Tobie ich przeglądania (i tworzenia kodu)! Jeśli jednak nie zamierzasz korzystać z aplikacji Jupyter Notebook, przekonwertowałem te notatniki również do postaci standardowych plików skryptowych Pythona (w formacie `.py`), które można przeglądać i edytować w dowolnym edytorze tekstu. 
