Python. Uczenie maszynowe - kod źródłowy


##  Rozdział 15. Klasyfikowanie obrazów za pomocą głębokich splotowych sieci neuronowych


### Spis treści

- Podstawowe elementy splotowej sieci neuronowej
  - Splotowe sieci neuronowe i hierarchie cech
  - Splot dyskretny
    - Splot dyskretny w jednym wymiarze
    - Uzupełnianie zerami jako sposób kontrolowania rozmiaru wyjściowych map cech
    - Określanie rozmiaru wyniku splotu
    - Splot dyskretny w dwóch wymiarach
  - Warstwy podpróbkowania
  - Implementowanie sieci CNN
    - Praca z wieloma kanałami wejściowymi/barw
    - Regularyzowanie sieci neuronowej metodą porzucania
  - Funkcje straty w zadaniach klasyfikacji
- Implementacja głębokiej sieci splotowej za pomocą biblioteki TensorFlow
  - Architektura wielowarstwowej sieci CNN
  - Wczytywanie i wstępne przetwarzanie danych
  - Implementowanie sieci CNN za pomocą interfejsu Keras
    - Konfigurowanie warstw sieci splotowej w interfejsie Keras
    - Konstruowanie sieci splotowej za pomocą interfejsu Keras
  - Klasyfikowanie płci na podstawie zdjęć twarzy za pomocą sieci splotowej
    - Wczytywanie zestawu danych CelebA
    - Przekształcanie obrazów i dogenerowanie danych
    - Uczenie modelu CNN jako klasyfikatora płci
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

**(nawet jeśli nie zamierzasz instalować aplikacji Jupyter Notebook, możesz przeglądać notatniki w serwisie GitHub. Wystarczy je kliknąć: [`r15_cz1.ipynb`](r15_cz1.ipynb) i [`r15_cz2.ipynb`](r15_cz2.ipynb))**

Oprócz samego kodu źródłowego, dołączyłem również w każdym notatniku Jupyter spis treści, a także nagłówki sekcji, które są spójne z treścią książki. Ponadto umieściłem również występujące w niej rysunki, dzięki czemu powinno Ci się łatwiej przeglądać zawartość plików i pracować z kodem.

![](../r02/rysunki/jupyter-przyklad-2.png)


Tworząc te notatniki przyświecał mi cel jak największego ułatwienia Tobie ich przeglądania (i tworzenia kodu)! Jeśli jednak nie zamierzasz korzystać z aplikacji Jupyter Notebook, przekonwertowałem te notatniki również do postaci standardowych plików skryptowych Pythona (w formacie `.py`), które można przeglądać i edytować w dowolnym edytorze tekstu.  
