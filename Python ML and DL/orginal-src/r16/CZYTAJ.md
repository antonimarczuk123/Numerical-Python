Python. Uczenie maszynowe - kod źródłowy


##  Rozdział 16. Modelowanie danych sekwencyjnych za pomocą rekurencyjnych sieci neuronowych


### Spis treści

- Wprowadzenie do danych sekwencyjnych
  - Modelowanie danych sekwencyjnych — kolejność ma znaczenie
  - Przedstawianie sekwencji
  - Różne kategorie modelowania sekwencji
- Sieci rekurencyjne służące do modelowania sekwencji
  - Mechanizm zapętlania w sieciach rekurencyjnych
  - Obliczanie aktywacji w sieciach rekurencyjnych
  - Rekurencja w warstwie ukrytej a rekurencja w warstwie wyjściowej
  - Problemy z uczeniem długofalowych oddziaływań
  - Jednostki LSTM
- Implementowanie wielowarstwowej sieci rekurencyjnej przy użyciu biblioteki TensorFlow do modelowania sekwencji
  - Pierwszy projekt — przewidywanie sentymentów na recenzjach z zestawu danych IMDb
    - Przygotowanie danych recenzji
    - Warstwy wektorów właściwościowych w kodowaniu zdań
    - Budowanie modelu sieci rekurencyjnej
    - Uczenie modelu sieci rekurencyjnej przeznaczonego do analizy sentymentów
  - Drugi projekt — modelowanie języka na poziomie znaków w TensorFlow
    - Przygotowanie danych
    - Tworzenie modelu sieci RNN przetwarzającej znaki
    - Faza ewaluacji: generowanie nowych fragmentów tekstu
- Przetwarzanie  języka za pomocą modelu transformatora
  - Wyjaśnienie mechanizmu samouwagi
  - Podstawowa wersja mechanizmu samouwagi
  - Parametryzowanie mechanizmu samouwagi za pomocą wag kwerendy, klucza i wartości
  - Wieloblokowy mechanizm uwagi i komórka transformatora
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

**(nawet jeśli nie zamierzasz instalować aplikacji Jupyter Notebook, możesz przeglądać notatniki w serwisie GitHub. Wystarczy je kliknąć: [`r16_cz1.ipynb`](r16_cz1.ipynb) i [`r16_cz2.ipynb`](r16_cz2.ipynb))**

Oprócz samego kodu źródłowego, dołączyłem również w każdym notatniku Jupyter spis treści, a także nagłówki sekcji, które są spójne z treścią książki. Ponadto umieściłem również występujące w niej rysunki, dzięki czemu powinno Ci się łatwiej przeglądać zawartość plików i pracować z kodem.

![](../r02/rysunki/jupyter-przyklad-2.png)


Tworząc te notatniki przyświecał mi cel jak największego ułatwienia Tobie ich przeglądania (i tworzenia kodu)! Jeśli jednak nie zamierzasz korzystać z aplikacji Jupyter Notebook, przekonwertowałem te notatniki również do postaci standardowych plików skryptowych Pythona (w formacie `.py`), które można przeglądać i edytować w dowolnym edytorze tekstu.
