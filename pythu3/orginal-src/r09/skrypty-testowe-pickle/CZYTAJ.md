**Uwaga**

Podrozdział dotyczący serializacji może być nieco zawiły, dlatego umieściłem tu prostsze skrypty testowe (katalog skrypty-testowe-pickle/) sprawdzające, czy Twoje środowisko pracy jest prawidłowo skonfigurowane. W zasadzie mamy tu do czynienia z uproszczoną wersją przykładów opisanych w rozdziale 8., a także z bardzo małym wycinkiem zestawu danych IMdB.

Uruchomienie skryptu

    python pickle-dump-test.py

spowoduje wyuczenie niewielkiego modelu klasyfikacji z pliku `filmy_dane_niewielki.csv` i utworzenie dwóch plików:

    stopwords.pkl
    classifier.pkl

Jeśli teraz uruchomisz skrypt

    python pickle-load-test.py

powinieneś otrzymać w wyniku dwa następujące wiersze:

    Prognoza: pozytywna
    Prawdopodobieństwo: 85.71%