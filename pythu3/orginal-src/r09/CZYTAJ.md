Python. Uczenie maszynowe - kod źródłowy

## Rozdział 9. Wdrażanie modelu uczenia maszynowego do aplikacji sieciowej

- Serializacja wyuczonych estymatorów biblioteki scikit-learn
- Konfigurowanie bazy danych SQLite
- Tworzenie aplikacji sieciowej za pomocą środowiska Flask
- Nasza pierwsza aplikacja sieciowa
  - Sprawdzanie i wyświetlanie formularza
  - Przekształcanie klasyfikatora recenzji w aplikację sieciową
- Umieszczanie aplikacji sieciowej na publicznym serwerze
  - Aktualizowanie klasyfikatora recenzji filmowych
- Podsumowanie

---

Kod aplikacji napisanych w środowisku Flask znajduje się w następujących katalogach:

- `flask_pierwsza_aplikacja_1/`: prosta aplikacja sieciowa napisana we Flasku,
- `flask_pierwsza_aplikacja_2/`: aplikacja `flask_pierwsza_aplikacja_1` z dodanym sprawdzaniem i wyświetlaniem formularza,
- `klasyfikator_filmowy/`: klasyfikator recenzji zagnieżdżony w aplikacji sieciowej,
- `zaktualizowany_klasyfikator_filmowy/`: to samo, co `klasyfikator_filmowy` lecz z dodaną funkcją aktualizowania bazy danych w momencie uruchomienia.


Aby uruchomić lokalnie aplikacje sieciowe, przejdź za pomocą polecenia `cd` do katalogu zawierającego powyższe aplikacje i uruchom główny skrypt, np.:

    cd ./flask_pierwsza_aplikacja_1
    python3 app.py

Teraz w terminalu powinien pojawić się następujący tekst:

     * Running on http://127.0.0.1:5000/
     * Restarting with reloader

Otwórz następnie przeglądarkę i wpisz adres wyświetlony w terminalu (najczęściej http://127.0.0.1:5000/) aby przejrzeć zawartość aplikacji.


**Odnośnik do działającego przykładu (w języku angielskim) utworzonego przy użyciu informacji zawartych w rozdziale: http://raschkas.pythonanywhere.com/**.