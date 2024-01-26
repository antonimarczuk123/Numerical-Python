Python. Uczenie maszynowe - kod źródłowy


##  Rozdział 6. Najlepsze metody oceny modelu i strojenie parametryczne

### Spis treści

- Usprawnianie cyklu pracy za pomocą kolejkowania
  - Wczytanie zestawu danych Breast Cancer Wisconsin
  - Łączenie funkcji transformujących i estymatorów w kolejce czynności
- Stosowanie k-krotnego sprawdzianu krzyżowego w ocenie skuteczności modelu
  - Metoda wydzielania
  - K-krotny sprawdzian krzyżowy
- Sprawdzanie algorytmów za pomocą krzywych uczenia i krzywych walidacji
  - Diagnozowanie problemów z obciążeniem i wariancją za pomocą krzywych uczenia
  - Rozwiązywanie problemów przetrenowania i niedotrenowania za pomocą krzywych walidacji
- Dostrajanie modeli uczenia maszynowego za pomocą metody przeszukiwania siatki
  - Strojenie hiperparametrów przy użyciu metody przeszukiwania siatki
  - Dobór algorytmu poprzez zagnieżdżony sprawdzian krzyżowy
- Przegląd wskaźników oceny skuteczności
  - Odczytywanie macierzy pomyłek
  - Optymalizacja precyzji i pełności modelu klasyfikującego
  - Wykres krzywej ROC
  - Wskaźniki zliczające dla klasyfikacji wieloklasowej
- Kwestia dysproporcji klas
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

**(nawet jeśli nie zamierzasz instalować aplikacji Jupyter Notebook, możesz przeglądać notatniki w serwisie GitHub. Wystarczy je kliknąć: [`r06.ipynb`](r06.ipynb))**

Oprócz samego kodu źródłowego, dołączyłem również w każdym notatniku Jupyter spis treści, a także nagłówki sekcji, które są spójne z treścią książki. Ponadto umieściłem również występujące w niej rysunki, dzięki czemu powinno Ci się łatwiej przeglądać zawartość plików i pracować z kodem.

![](../r02/rysunki/jupyter-przyklad-2.png)


Tworząc te notatniki przyświecał mi cel jak największego ułatwienia Tobie ich przeglądania (i tworzenia kodu)! Jeśli jednak nie zamierzasz korzystać z aplikacji Jupyter Notebook, przekonwertowałem te notatniki również do postaci standardowych plików skryptowych Pythona (w formacie `.py`), które można przeglądać i edytować w dowolnym edytorze tekstu. 