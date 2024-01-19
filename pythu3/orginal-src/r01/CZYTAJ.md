Sebastian Raschka, 2019

Python. Uczenie maszynowe - kod źródłowy


##  Rozdział 1. Umożliwianie komputerom uczenia się z danych


---

**Rozdział 1. nie zawiera listingów.**

---

## Instalacja pakietów w środowisku Python

Omawiane środowisko programistyczne jest dostępne na wszystkie główne systemy operacyjne — Microsoft Windows, Mac OS X i Linuksa — a zarówno jego instalator, jak i dokumentację znajdziesz na oficjalnej stronie Pythona: https://www.python.org.

Ta książka została napisana pod kątem Pythona w wersji co najmniej `>= 3.7.0`, natomiast zalecam korzystanie z najbardziej aktualnej implementacji tego środowiska (w wersji 3.), chociaż większość przykładowych kodów powinna być również kompatybilna z wersją `>= 2.7.10`. Jeżeli postanowisz uruchamiać zawarte w książce kody w wersji 2.7 Pythona, zapoznaj się najpierw z głównymi różnicami pomiędzy obydwiema wersjami środowiska programowania. Dobre podsumowanie różnic pomiędzy wersjami 2.7 i 3. Pythona (w języku angielskim) znajdziesz pod adresem https://wiki.python.org/moin/Python2orPython3.

**Uwaga**

Możesz sprawdzić bieżącą, domyślną wersję środowiska Python za pomocą polecenia:

    $ python -V

W moim przypadku zwracana jest informacja:

    Python 3.7.1 :: Continuum Analytics, Inc.


#### Pip

Dodatkowe, wykorzystywane w dalszej części książki pakiety można zainstalować za pomocą aplikacji `pip` stanowiącej część standardowej biblioteki Pythona od wersji 3.3. Więcej informacji (w języku angielskim) na temat instalatora pip znajdziesz pod adresem https://docs.python.org/3/installing/index.html.

Po zainstalowaniu środowiska Python dodajemy kolejne pakiety wpisując następującą komendę w wierszu poleceń:

    pip install JakiśPakiet


(gdzie w miejsce wyrażenia `JakiśPakiet` wstawiamy nazwę pakietu numpy, pandas, matplotlib, scikit-learn itd.).

Zainstalowane pakiety możemy zaktualizować za pomocą flagi  `--upgrade`:

    pip install JakiśPakiet --upgrade


#### Anaconda

Bardzo polecaną, alternatywną dystrybucją Pythona przeznaczoną do obliczeń naukowych jest Anaconda stworzona przez firmę Continuum Analytics. Jest to bezpłatna dystrybucja — również w przypadku zastosowań komercyjnych — zawierająca wszystkie niezbędne pakiety wykorzystywane w analizie danych, obliczeniach matematycznych oraz inżynierii, dostępne w przyjaznej, międzyplatformowej postaci. Instalator Anacondy znajdziesz pod adresem https://docs.anaconda.com/anaconda/install/, z kolei szybkie wprowadzenie do tego środowiska jest dostępne na stronie https://docs.anaconda.com/anaconda/user-guide/getting-started/.

Po zainstalowaniu Anacondy możemy instalować nowe pakiety Pythona za pomocą następującego polecenia:

    conda install JakiśPakiet

Zainstalowane pakiety aktualizujemy korzystając z poniższej komendy:

    conda update JakiśPakiet

Przez większość czasu będziemy korzystać z wielowymiarowych tablic biblioteki NumPy do przechowywania i przetwarzania danych. Sporadycznie zastosujemy również bibliotekę pandas — nakładkę biblioteki NumPy zapewniającą dodatkowe, zaawansowane narzędzia do manipulowania danymi, dzięki czemu praca z tabelarycznymi informacjami będzie jeszcze wygodniejsza. Aby usprawnić proces nauki i zwizualizować dane ilościowe (pozwala to w maksymalnie intuicyjny sposób zrozumieć wykonywane działania), wprowadzimy również do użytku wysoce konfigurowalną bibliotekę matplotlib.

#### Najważniejsze biblioteki

Poniżej wymieniam numery wersji głównych pakietów Pythona, które były wykorzystane w trakcie pisania niniejszej książki. Upewnij się, że masz na swoim komputerze zainstalowane przynajmniej te wersje (lub nowsze), dzięki czemu przykładowe kody będą działały we właściwy sposób:

- [NumPy](http://www.numpy.org) >= 1.17.4
- [SciPy](http://www.scipy.org) >= 1.3.1
- [scikit-learn](http://scikit-learn.org/stable/) >= 0.22.0
- [matplotlib](http://matplotlib.org) >= 3.1.0
- [pandas](http://pandas.pydata.org) >= 0.25.3

## Python/Jupyter Notebook

Dla niektórych czytelników zagadkę stanowił format `.ipynb` przykładowych plików -- są to notatniki aplikacji IPython. Wybrałem je zamiast klasycznych skryptów `.py`, ponieważ uważam, że nadają się one znakomicie do projektów zajmujących się analizą danych! Notatniki aplikacji IPython pozwalają na umieszczanie wszystkiego w jednym miejscu: kodu źródłowego, wyników działania algorytmów, wykresów danych oraz dokumentacji obsługującej składnie przydatnego języka Markdown oraz potężnej platformy LaTeX!

![](./images/ipynb_ex1.png)

**Uwaga na marginesie:**  Aplikacja "IPython Notebook" została ostatnio przechrzczona na "[Jupyter Notebook](<http://jupyter.org>)"; Jupyter jest projektem parasolowym, w którym główny nacisk został położony na obsługę dodatkowych języków poza Pythonem, takich jak Julia, R i wiele innych. Jako użytkownik Pythona nie musisz się jednak obawiać - w Twoim przypadku różnice dotyczą wyłącznie terminologii (teraz zamiast aplikacji "Ipython Notebook" mówimy o programie "Jupyter Notebook").

Aplikację Jupyter notebook możemy, jak zwykle, zainstalować za pomocą programu pip:

    $ pip install jupyter notebook

Jeżeli korzystasz ze środowiska Anaconda lub Miniconda, możesz ewentualnie skorzystać z instalatora Conda:

    $ conda install jupyter notebook

Aby otworzyć notatnik Jupyter, przechodzimy (`cd`) do katalogu zawierającego przykładowe kody, np.:


    $ cd ~/kod/python-uczenie-maszynowe

a następnie uruchamiamy aplikację `jupyter notebook` w następujący sposób:

    $ jupyter notebook

Aplikacja Jupyter zostanie uruchomiona w domyślnej przeglądarce (najczęściej pod adresem [http://localhost:8888/](http://localhost:8888/)). Teraz wystarczy wybrać w menu aplikacji notatnik, który zamierzasz przeglądać.

![](./images/ipynb_ex2.png)

Więcej informacji na temat aplikacji Jupyter Notebook (w języku angielskim) znajdziesz w [elementarzu aplikacji Jupyter](http://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/what_is_jupyter.html), a także w [podstawach notatników Jupyter](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html).

## Jupyter Lab

Alternatywnym rozwiązaniem jest używanie stworzonego w 2018 r. narzędzia Jupyter Lab. Współpracuje z takimi samymi typami plików `.ipynb`, ale zawiera dodatkowe funkcje interfejsu przeglądarki. Nie ma różnicy, z której aplikacji korzystasz, natomiast możesz zainstalować Jupyter Lab w następujący sposób: 

    $ conda install -c conda-forge jupyterlab
    
Z kolei uruchamiamy tę aplikację za pomocą polecenia: 

    $ jupyter lab
    
W ten sposób zostanie uruchomiona sesja Jupyter Lab w przeglądarce. Więcej informacji na temat projektu Jupyter Lab znajdziesz w oficjalnej dokumentacji: https://jupyterlab.readthedocs.io/en/stable/,
