Dokument Wymagań Produktowych (PRD) - Moduł Klienta TMS

1. Cel Produktu

Stworzenie intuicyjnego, nowoczesnego interfejsu webowego (MVP), który pozwoli klientom firmy transportowej na samodzielne składanie zamówień transportowych. Celem jest eliminacja błędów komunikacyjnych (e-mail/telefon) i standaryzacja danych wejściowych.

2. Grupa Docelowa

Logistycy po stronie klienta (B2B).

Pracownicy biurowi zlecający transporty ad-hoc.

3. Zakres Implementacji Frontendowej (MVP)

Niniejsza sekcja skupia się wyłącznie na warstwie prezentacji i logice UI. Dane powinny być zarządzane w stanie lokalnym (np. React State / Pinia).

Kluczowe funkcjonalności UI:

Multi-step Wizard Form: Podział procesu na logiczne etapy.

Walidacja w czasie rzeczywistym: Informowanie o błędach przed przejściem do kolejnego kroku.

Podsumowanie zamówienia: Widżet "Sticky" z boku ekranu pokazujący postęp i wybrane dane.

Responsywność: Interfejs dostosowany do desktopów i tabletów.

4. Szczegółowy Proces: Formularz "Nowe Zamówienie"

Proces został podzielony na 4 kroki: Trasa, Ładunek, Wymagania Specjalne, Podsumowanie.

Krok 1: Trasa i Terminy

Miejsce Załadunku:

Input: Nazwa firmy / Kontakt.

Input: Adres (Ulica, Kod pocztowy, Miasto, Kraj - dropdown).

Date Picker: Data i preferowane okno godzinowe.

Miejsce Rozładunku:

Analogiczne pola jak powyżej.

Logic: Walidacja, czy data rozładunku nie jest wcześniejsza niż załadunku.

Krok 2: Specyfikacja Ładunku

Typ Jednostki: Dropdown (Paleta Euro, Paleta Przemysłowa, LDM, Luzem).

Liczba sztuk: Input numeryczny.

Wymiary: Długość, Szerokość, Wysokość (cm).

Waga: Całkowita waga brutto (kg).

Czy towar niebezpieczny (ADR)?: Toggle/Checkbox.

Jeśli TAK: Dodatkowy input na kod UN i klasę.

Wartość towaru: Input walutowy (opcjonalne dla ubezpieczenia).

Krok 3: Wymagania i Usługi Dodatkowe

Typ Nadwozia: Multiselect (Plandeka, Chłodnia, Izoterma, Mega).

Wyposażenie dodatkowe: Checkboxy (Winda, Paleciak, Pasy, Narożniki).

Uwagi dla kierowcy: Pole Textarea.

Krok 4: Podsumowanie i Akceptacja

Wyświetlenie wszystkich danych w formie czytelnej karty.

Pole na numer referencyjny klienta (PO Number).

Akceptacja regulaminu (Checkbox).

Przycisk "Złóż zamówienie" (Symulacja wysłania).

5. Wytyczne Techniczne dla Frontendu

UI / UX

Framework: React.js / Vue.js.

Stylizacja: Tailwind CSS lub Styled Components dla szybkiego budowania interfejsu.

Ikony: Lucide React lub FontAwesome (ikony ciężarówek, kalendarza, paczek).

Pasek Postępu (Stepper): Widoczny na górze strony, pokazujący aktualny etap.

Logika Formularza

Stan (State): Jeden obiekt przechowujący dane z wszystkich kroków.

Walidacja:

Pola wymagane oznaczone gwiazdką *.

Blokada przycisku "Dalej", jeśli krok nie jest poprawnie wypełniony.

Local Storage: (Opcjonalnie) Zapisywanie draftu zamówienia w przeglądarce, aby nie stracić danych przy odświeżeniu.

Przykład Struktury Obiektu Danych:

{
  "sender": {
    "company": "Logistics Hub SP. z o.o.",
    "address": "ul. Transportowa 1, 00-001 Warszawa",
    "date": "2024-05-20",
    "time_window": "08:00-16:00"
  },
  "receiver": {
    "company": "Klienci Sp. k.",
    "address": "ul. Dostawcza 5, 30-001 Kraków",
    "date": "2024-05-21",
    "time_window": "07:00-14:00"
  },
  "cargo": {
    "type": "Euro Pallet",
    "count": 4,
    "weight": 1200,
    "adr": false
  },
  "requirements": {
    "truck_type": "Tautliner",
    "lift_required": true
  }
}


6. Przypadki Testowe (Frontend)

Czy przycisk "Wstecz" zachowuje wpisane dane?

Czy walidator kodu pocztowego reaguje na format kraju (np. PL: 00-000)?

Czy po zaznaczeniu ADR pojawiają się dodatkowe wymagane pola?

Czy formularz poprawnie przelicza łączną wagę, jeśli dodawanych jest kilka typów towaru?