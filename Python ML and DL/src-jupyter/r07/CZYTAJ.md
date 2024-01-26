Python. Uczenie maszynowe - kod źródłowy


##  Rozdział 7. Łączenie różnych modeli w celu uczenia zespołowego

### Spis treści

- Uczenie zespołów
- Łączenie klasyfikatorów za pomocą algorytmu głosowania większościowego
  - Implementacja prostego klasyfikatora głosowania większościowego
  - Stosowanie reguły głosowania większościowego do uzyskiwania prognoz
  - Ewaluacja i strojenie klasyfikatora zespołowego
- Agregacja — tworzenie zespołu klasyfikatorów za pomocą próbek początkowych
  - Agregacja w pigułce
  - Stosowanie agregacji do klasyfikowania przykładów z zestawu Wine
- Usprawnianie słabych klasyfikatorów za pomocą wzmocnienia adaptacyjnego
  - Wzmacnianie — mechanizm działania
  - Stosowanie algorytmu AdaBoost za pomocą biblioteki scikit-learn
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

**(nawet jeśli nie zamierzasz instalować aplikacji Jupyter Notebook, możesz przeglądać notatniki w serwisie GitHub. Wystarczy je kliknąć: [`r07.ipynb`](r07.ipynb))**

Oprócz samego kodu źródłowego, dołączyłem również w każdym notatniku Jupyter spis treści, a także nagłówki sekcji, które są spójne z treścią książki. Ponadto umieściłem również występujące w niej rysunki, dzięki czemu powinno Ci się łatwiej przeglądać zawartość plików i pracować z kodem.

![](../r02/rysunki/jupyter-przyklad-2.png)


Tworząc te notatniki przyświecał mi cel jak największego ułatwienia Tobie ich przeglądania (i tworzenia kodu)! Jeśli jednak nie zamierzasz korzystać z aplikacji Jupyter Notebook, przekonwertowałem te notatniki również do postaci standardowych plików skryptowych Pythona (w formacie `.py`), które można przeglądać i edytować w dowolnym edytorze tekstu.
