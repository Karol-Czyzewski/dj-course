# Zadanie 1

Stwórz setup observability dla deliveroo/Customer Portal
- codebase: `M8/customer-portal`
- stack: nuxt/vue (server-side frontend, _poniekąd_ w stylu "good old PHP")

**ZADANIE**:
- skonfigurować infrastrukturę (konkretne komponenty dockerowe, ich configi, poprawna komunikacja między elementami)
- zinstrumentować kod apki: apka wysyła METRYKI i LOGI
- w grafanie tworzysz dashboardy z podstawowymi wizualizacjami
- na poziomie infry dodajesz również [`node-exporter`](https://hub.docker.com/r/prom/node-exporter/) (który wysyła diagnostyczne dane telemetryczne dot. node'a w sensie **maszyny**, _nie nodejs_)
- na poziomie infry konfigurujesz dodatkowo standardowy mongodb monitoring

**CEL**: kiedy przeklikujesz aplikację, w grafanie/dashboardach widzisz adekwatnie cały ruch

Pochwal się na discordzie zrzutem grafany ilustrującym ruch w CP.

# Zadanie 2

Napraw rozwalony setup 👊:
- setup/codebase: `M9/o11y-homework`
- działamy tylko w ramach (grafana UI): `Home > Dashboards > (OTLP) Metrics Dashboard`. Tylko ten dashboard - ale i cały setup telemetry który musi działać aby go nakarmić
- zostało celowo wstawione **kilka** prostych, ale **trudnych do znalezienia** błędów. Ręcznie byłaby to masakra - a z AI powinno pójść relatywnie szybko.
- **BŁĘDY mogą dotyczyć każdej płaszczyzny**: instrumentacji, konfiguracji dockera, konfiguracji w yamlach (narzędzi telemetrycznych nie-dockerowych), czy samej grafany - jej konfiga albo i konfigów samych dashboardów.
- Naprawianie błędów będzie etapami. Jeśli komunikacja (flow danych telemetrycznych) leży, to może apka ich nie wysyła? A może Otel Collector ich nie odbiera? A jak to naprawisz - i dane będą np. dostępne w storage'u typu prometheus - to może grafana nie może się do nich dobrać? Albo grafana dane widzi, ale dashboard jest niepoprawnie skonfigurowany?

Wyzwanie/pytanie: 
- co zlecać modelowi - a co robić samodzielnie?
- jak bardzo rozbić task na mniejsze, aby nie "zeżreć" ogromnej ilości tokenów?
- storage telemetryczny trzeba nakarmić danymi (trzeba wygenerować jakiś ruch - "up to you" jak to zrobisz)

**ZADANIE**:
- wszystkie widgety w dashboardzie `(OTLP) Metrics Dashboard` pokazują dane - i robią to poprawnie.

**CEL**: “zaradność” w szybkim/skutecznym rozwiązywaniu potencjalnie skomplikowanych problemów, przy jednoczesnym intensywnym wspomaganiu się AI przy troubleshootingu. Umiesz zidentyfikować **kilka** problemów i je naprawdić. Jednocześnie, nakierowujesz model zarówno pod kątem celu jaki chcesz osiągnąć (naprawić setup) jak i weryfikacji tego co demo robi po drodze.

Koniecznie pochwal się na discordzie :) ale nie konkretnym punktem (np. która linijka kodu, port, opcja konfiguracyjna czy cokolwiek) - tylko opisowo - trzeba było naprawić (ogólnie) np. flow danych między X a Y lub dashboard itp. I wrzuć screena z dashboardem z danymi (co dowiedzie że wszystko zostało odetkane).

# Zadanie 3

Diagnostyka/Debugowanie setupu o11y z AI.
Każ modelowi “odpytać” Prometheusa:
- codebase/setup: `M9/o11y-metrics` lub `M9/o11y-full`
- wygeneruj “trochę trafficu” w tym setpie (np. przy użyciu stress tests, oczywiście po uprzednim postawieniu apki w dockerze).

**ZADANIE**: Każ modelowi sprawdzić, czy metryki odnotowały ruch. Model powinien krok po kroku sprawdzać (a jeśli tego nie robi to nakaż mu to explicite) aby wykonywał sekwencję:
- ręczne wysłanie żądania do usługi (request HTTP do serwera w wybranym setupie)
- model z kodu wie, jakie dane telemetryczne powinny być wysyłane
- sprawdza w API REST prometheusa, czy są tam dostępne (a jeśli jest pośrednik w postaci open telemetry collector - to sprawdza najpierw tam)
- jeśli są w prometheusie, to potem sprawdza czy query grafanowe te dane widzi
- na samym końcu sprawdza, czy konfiguracja konkretnego dashboarda (tego którego oglądasz Ty) uruchamia odpowiednie query.
Cała ta sekwencja odzwierciedla, co się dzieje pod maską - i pozwala zdiagnozować gdzie i co jest "urwane".

**CEL**: Umiesz wykorzystać LLMy aby te, korzystając z przeróżnych API wystawianych przez prometheuse, grafany etc błyskawicznie diagnozowały - czy ruch danych telemetrycznych przebiega prawidłowo.
(long story short: robienie tego ręcznie byłoby wybitnie czasochłonne i błędogenne, tu AI sprawdza się wyśmienicie)

Wrzuć na discorda zrzuty jak agent radzi sobie z szybkim montowaniem mega niskopoziomowym potoków żądań i późniejszej ich przeróbki (coś co Tobie/mi zajęłoby duuuużo czasu)

# Zadanie 4

Stwórz nowy dashboard!
- setup/codebase: `M9/o11y-full`
- nakarm swoje instancje (dockerowe) danymi telemetrycznymi poprzez np. uruchomienie stress testów (`M9/load-testing-artillery` lub `M9/load-testing-taurus`)
- stwórz nowy dashboard w grafanie

ZADANIE. Dashboard ma zawierać:
- **Histogramy**: `http_client_duration_milliseconds` oraz `http_server_duration_milliseconds`
- tempo przyrastania **logów BŁĘDÓW**
- **Top N** najdłużej trwających requestów HTTP

**CEL**: Umiesz stworzyć od zera grafanowy dahboard który wizualizuje metryki o średnim poziomie skomplikowania

# Zadanie 5

(kontynuacja zadania 4)

Dodaj metrykę mierząca liczbę połączeń w puli
- do wyboru: postgres albo mongo
- widoczna w dashboardzie

# Zadanie 6

(kontynuacja zadania 4+5)

Zmienne w UI dashboardu grafanowego:
- stwórz nowy (lub wykorzystaj z wcześniejszego zadania 4/5)
- dodaj dowolny widget (input, dropdown, etc.)
- skonfiguruj klikalną zmienną (np. `job`, `url`, `method`, cokolwiek). Możesz wylistować dostępne wartości - lub pobrać istniejące wartości z danych dostępnych np. w metrykach (np. na jakie URLe przychodziły ządania, jakie były odpowiedzi, itp).
- sprawdź poprawność konfiguracji - po zmianie wartości dashboard powinien wyświetlać adekwatne dane również wtedy, gdy widget jest pusty (pusty input, brak wybranej opcji w dropdownie etc)

**CEL**: umiesz "zdynamizować" swoje dashboardy dzięki klikalnym zmiennym (pola tesktowe, dropdown, etc.). W dłuszej perspektywie jednak liczy się to - czy wiesz CO dynamizować i PO CO - czyli JAKICH danych będziecie w dashboardzie szukać :)

Pochwal się zrzutami jak UI grafany dostosowuje się do zmiany zmiennej.

# Zadanie 7

Nowy scenariusz stress testowy
- wybierasz tool (taurus, artillery lub cokolwiek chcesz)

ZADANIE:
- scenariusz stress testowy ma trwa długo i ma “fluktuować” (zwiększa się ruch, potem zmniejsza, potem znowu zwiększa jeszcze bardziej itp). Ma być rozbudowany i mieć kilka faz.
- Uruchom na dowolnym setupie z `M9`
- sprawdź czy ruch jest odnotowany w grafanie - i czy odzwierciedla charakter testu, spike'i, spadki, itp.

Ostatecznie:
- rozpocząłeś w toolu X, to teraz każ LLMowi przemigrować na tool Y
- sprawdź czy test przemigrowany (ten nowy) ma analogiczną logikę/charakterystykę co pierwowzór

Pochwal się zrzutami dashboardu :) pochwal się podsumowaniem scenariusza.

# Zadanie 8

Troubleshooting w setupie `o11y-full`.

Kontekst:
- mamy błąd który objawia się tym, że podczas uruchamiania stress testu _massive_ w dashboardzie NIE są wyświetlane błędy
- czy problem jest w instrumentacji? Czy w komunikacji pomiędzy kontenerami? A moe w konfigu dashbaordu grafany? Może nie to query co trzeba? Któż to wie 🫠

**ZADANIE**:
- zidentyfikować i poprawić błąd :) po uruchomieniu stress testu _massive_ błędy powinny być widoczne w grafanie
- oczywiście, ramię w ramię z modelem

**CEL**: Potrafisz wystarczająco precyzyjnie opisać modelowi problem, a następnie orkiestrować/nazdorować model i nakierowywać go na znajdowanie problemu. Rozumiesz infrastrukturę/instrumentację/flow danych itp itd - wszystko co jest pod spodem - i potrafisz orzec czy model idzie we właściwym kierunku, czy niekoniecznie (i trzeba mu pomóc).
