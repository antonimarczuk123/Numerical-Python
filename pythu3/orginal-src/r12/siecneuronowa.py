import numpy as np
import sys


class NeuralNetMLP(object):
    """ Sieć neuronowa z dodatnim sprzężeniem zwrotnym / klasyfikator – perceptron wielowarstwowy.

    Parametry
    ------------
    n_hidden : liczba całkowita (domyślnie: 30)
          Liczba jednostek ukrytych.
    l2 : wartość zmiennoprzecinkowa (domyślnie: 0.0)
          Parametr lambda regularyzacji L2.
          Nie ma regularyzacji, jeśli l2=0.0 (domyślnie)
    epochs : liczba całkowita (domyślnie: 500)
          Liczba przebiegów po zestawie danych uczących.
    eta : wartość zmiennoprzecinkowa (domyślnie: 0.001)
          Współczynnik uczenia.
    shuffle : typ boolowski (domyślnie: True)
          Jeżeli wartość jest równa "True", tasuje dane uczące po każdej epoce
          w celu uniknięcia cykliczności.
    minibatch_size : liczba całkowita (domyślnie: 1)
          Liczba przykładów uczących na daną mini-grupę.
    seed : liczba całkowita (domyślnie: None)
          Ziarno losowości używane podczas tasowania i inicjalizowania wag.

    Atrybuty
    -----------
    eval_: słownik
          Słownik przechowujący wartości kosztu, dokładności uczenia i dokładności
          walidacji dla każdej epoki podczas uczenia.

    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """Koduje etykiety do postaci "gorącojedynkowej"

        Parametry
        ------------
        y : tablica, wymiary = [n_przykładów]
            Wartości docelowe.
        n_classes : int
            Liczba klas.

        Zwraca
        -----------
        onehot : array, wymiary = (n_przykładów, n_etykiet)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Oblicza funkcję logistyczną (sigmoidalną)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Oblicza krok propagacji w przód"""

        # Etap 1.: Pobudzenie całkowite warstwy ukrytej
        # [n_przykładów, n_cech] dot [n_cech, n_hidden]
        # -> [n_przykładów, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # Etap 2.: Aktywacja warstwy ukrytej
        a_h = self._sigmoid(z_h)

        # Etap 3.: Całkowite pobudzenie warstwy wyjściowej
        # [n_przykładów, n_hidden] dot [n_hidden, n_etykiet_klas]
        # -> [n_przykładów, n_etykiet_klas]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # Etap 4.: Aktywacja warstwy wyjściowej
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """Oblicza funkcję kosztu.

        Parametry
        ----------
        y_enc : tablica,wymiary = (n_przykładów, n_etykiet)
            Etykiety klas zakodowane do postaci "gorącojedynkowej".
        output : tablica, wymiary = [n_przykładów, n_jednostek_wyjściowych]
            Aktywacja warstwy wyjściowej (propagacja w przód).

        Zwraca
        ---------
        cost : wartość zmiennoprzecinkowa
            Regularyzowana funkcja kosztu.

        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        
        # Jeżeli będziesz stosować tę funkcję kosztu do innych zestawów danych,
        # w których wartości aktywacji mogą przyjmować skrajniejsze
        # wartości (bliższe 0 lub 1), możesz natrafić na błąd
        # "ZeroDivisionError" z powodu niestabilności numerycznej
        # bieżących implementacji języka Python i biblioteki NumPy.
        # Przykładowo, algorytm stara się obliczyć log(0), czyli niezdefiniowaną wartość.
        # Aby rozwiązać ten problem, możesz dodać niewielką stałą
        # do wartości aktywacji; stała ta zostanie dodana do funkcji logarytmicznej.
        #
        # Na przykład:
        #
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)
        
        return cost

    def predict(self, X):
        """Prognozowanie etykiet klas

        Parametry
        -----------
        X : tablica, wymiary = [n_przykładów, n_cech]
            Warstwa wejściowa z pierwotnymi cechami.

        Zwraca:
        ----------
        y_pred : tablica, wymiary = [n_przykładów]
            Przewidywane etykiety klas.

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Aktualizuje wagi za pomocą danych uczących.

        Parametry
        -----------
        X_train : tablica, wymiary = [n_przykładów, n_cech]
            Warstwa wejściowa zawierająca pierwotne cechy.
        y_train : tablica, wymiary = [n_przykładów]
            Docelowe etykiety klas.
        X_valid : tablica, wymiary = [n_przykładów, n_cech]
            Przykładowe cechy służące do walidacji w trakcie uczenia.
        y_valid : tablica, wymiary = [n_przykładów]
            Przykładowe etykiety służące do walidacji w trakcie uczenia.

        Zwraca:
        ----------
        self

        """
        n_output = np.unique(y_train).shape[0]  # liczba etykiet klas
        n_features = X_train.shape[1]

        ########################
        # Inicjalizowanie wag
        ########################

        # wagi pomiędzy warstwą wejściową a ukrytą
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # wagi pomiędzy warstwą ukrytą a wyjściową
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # formatowanie postępów
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # przebiegi po epokach uczenia
        for i in range(self.epochs):

            # przebiegi po mini-grupach
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # propagacja w przód
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Propagacja wsteczna
                ##################

                # [n_przykładów, n_etykiet_klas]
                delta_out = a_out - y_train_enc[batch_idx]

                # [n_przykładów, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_przykładów, n_etykiet_klas] dot [n_etykiet_klas, n_hidden]
                # -> [n_przykładdów, n_hidden]
                delta_h = (np.dot(delta_out, self.w_out.T) *
                           sigmoid_derivative_h)

                # [n_cech, n_ przykładów] dot [n_przykładów, n_hidden]
                # -> [n_cech, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # [n_hidden, n_przykładów] dot [n_przykładów, n_etykiet_klas]
                # -> [n_hidden, n_ etykiet_klas]
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # Regularyzacja i aktualizowanie wag
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # obciążenie nie jest regularyzowane
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # obciążenie nie jest regularyzowane
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Ewaluacja
            #############

            # Ewaluacja po każdej epoce w trakcie uczenia
            z_h, a_h, z_out, a_out = self._forward(X_train)
            
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Koszt: %.2f '
                             '| Dokładność uczenia/walidacji: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self