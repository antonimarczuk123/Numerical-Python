Python. Uczenie maszynowe - kod źródłowy


##  Rozdział 17. Generatywne sieci przeciwstawne w zadaniach syntetyzowania nowych danych


### Spis treści

- Wprowadzenie do generatywnych sieci przeciwstawnych
  - Autokodery
  - Modele generatywne syntetyzujące nowe dane
  - Generowanie nowych prób za pomocą sieci GAN
  - Funkcje straty generatora i dyskryminatora w modelu GAN
- Implementowanie sieci GAN od podstaw
  - Uczenie modeli GAN w środowisku Google Colab
  - Implementacja sieci generatora i dyskryminatora
  - Definiowanie zestawu danych uczących
  - Uczenie modelu GAN
- Poprawianie jakości syntetyzowanych obrazów za pomocą sieci GAN: splotowej i Wassersteina
  - Splot transponowany
  - Normalizacja wsadowa
  - Implementowanie generatora i dyskryminatora
  - Wskaźniki odmienności dwóch rozkładów
  - Praktyczne stosowanie odległości EM w sieciach GAN
  - Kara gradientowa
  - Implementacja sieci WGAN-DP służącej do uczenia modelu DCGAN
  - Załamanie modu
  - Inne zastosowania modeli GAN
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

**(nawet jeśli nie zamierzasz instalować aplikacji Jupyter Notebook, możesz przeglądać notatniki w serwisie GitHub. Wystarczy je kliknąć: [`r17_cz1.ipynb`](r17_cz1.ipynb) i [`r17_cz2.ipynb`](r17_cz2.ipynb))**

Oprócz samego kodu źródłowego, dołączyłem również w każdym notatniku Jupyter spis treści, a także nagłówki sekcji, które są spójne z treścią książki. Ponadto umieściłem również występujące w niej rysunki, dzięki czemu powinno Ci się łatwiej przeglądać zawartość plików i pracować z kodem.

![](../r02/rysunki/jupyter-przyklad-2.png)


Tworząc te notatniki przyświecał mi cel jak największego ułatwienia Tobie ich przeglądania (i tworzenia kodu)! Jeśli jednak nie zamierzasz korzystać z aplikacji Jupyter Notebook, przekonwertowałem te notatniki również do postaci standardowych plików skryptowych Pythona (w formacie `.py`), które można przeglądać i edytować w dowolnym edytorze tekstu. 
