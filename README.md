Ten projekt wykorzystuje algorytm regresji logistycznej do przewidywania ryzyka kredytowego na podstawie niemieckich danych kredytowych. Model klasyfikuje kredyty jako "dobry" (niski poziom ryzyka) lub "zły" (wysoki poziom ryzyka) na podstawie cech klienta.

Funkcjonalności
Wstępne przetwarzanie danych (mapowanie wartości kategorycznych, usuwanie braków danych).
Skalowanie cech za pomocą StandardScaler.
Trenowanie modelu regresji logistycznej z optymalnymi hiperparametrami.
Ocena modelu za pomocą:
Dokładności (accuracy),
Raportu klasyfikacji (classification report).
Analiza ważności cech wpływających na ryzyko kredytowe.
Predykcja dla nowych klientów.
Dane wejściowe
Cechy klientów, takie jak wiek, płeć, cel kredytu, kwota kredytu, oszczędności, itp.
Wymagania
Python
Biblioteki: pandas, scikit-learn
Sposób działania
Wczytanie i wstępne przetwarzanie danych:

Usunięcie brakujących danych.
Zamiana wartości tekstowych na numeryczne (np. "male" → 0, "female" → 1).
Skalowanie cech:

Dane są skalowane za pomocą StandardScaler, aby zapewnić lepsze działanie modelu regresji logistycznej.
Trenowanie modelu:

Model regresji logistycznej jest trenowany na zbiorze treningowym.
Ocena modelu:

Obliczenie dokładności oraz wygenerowanie raportu klasyfikacji.
Analiza cech:

Obliczenie ważności cech, aby określić, które czynniki mają największy wpływ na ryzyko kredytowe.
Predykcja:

Przewidywanie ryzyka kredytowego na podstawie nowych danych wejściowych.
Wyniki
Dokładność modelu: Wyświetlana w konsoli.
Ważność cech: Analiza wpływu każdej cechy na decyzję modelu.
Przykładowe predykcje: Model przewiduje, czy kredyt jest "dobry" czy "zły" dla nowych przypadków.
