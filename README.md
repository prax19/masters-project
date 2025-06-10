# Segmentacja semantyczna w systemach nowoczesnych pojazdów

Niniejsze repozytorium zawiera projekt magisterski realizowany w ramach pracy pt. **Zastosowanie sztucznych sieci neuronowych w systemach pojazdów autonomicznych** autorstwa **Patryka Piróg**, realizowanej w ramach grupy badawczej **ZSDiiZ** (Uniwersytet Śląski w Katowicach).

Projekt skupia się wokół zastosowania technik uczenia głębokiego w treningu modeli realizujących zadanie *segmentacji semantycznej*. Głównym celem projektu jest przygotowanie skryptów w języku *Python*, które posłużą do **treningu**, **ewaluacji** oraz **inferencji** modeli konwolucyjnych w celu porównania ich wydajności.

## Porównywane architektury
- [**resnet50**](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) ze wstępnie wytrenowanymi wagami.

## Testowane zbiory danych
- [**A2D2**](https://www.a2d2.audi/en/) - zbiór danych grupy VAG, składający się z wielu scen uchwyconych w niemieckich miastach. W skład zbioru wchodzą obrazy `.png` z wielu kamer w rozdzielczości *1920x1208*, dane z *magistrali CAN* oraz *LIDARu*.
- [**Cityscapes**](https://www.cityscapes-dataset.com/) - bardzo popularny zbiór danych w rozważaniu problemu segmentacji, uchwycony na terenie wielu niemieckich miast. Zbiór ten charakteryzuje się bardzo dużą różnorodnością oraz nieco zredukowaną ilością klas (nie obsługuje między innymi oznaczeń poziomych na drodze).

## Instrukcja uruchomienia
1. Zainstalować zależności ``pip install -r requirements.txt``.
2. Pobrać ręcznie [pełny zbiór danych A2D2](https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic.tar), bądź Cityscapes ([obrazy](https://www.cityscapes-dataset.com/file-handling/?packageID=3) oraz [maski](https://www.cityscapes-dataset.com/file-handling/?packageID=1)).
3. Rozpakować zbiory danych z użyciem oprogramowania archiwizującego jak np. [7-zip](https://www.7-zip.org/).
4. Przenieść rozpakowany folder z każdego archiwum do odpowiednich katalogów wewnątrz `./data/`.

### FAQ przy instalacji
- Czy lista `requirements.txt` jest kompletna?

    Lista może nie być kompletna. Zalecane jest ręczne doinstalowanie brakujących zależności.

- Jak w zasadzie powinna wyglądać struktura plików w katalogu `./data/`?

    W katalogu `./data/` znajdują się foldery `./data/a2d2` oraz `./data/cityscapes`. Są to katalogi przygotowane dla aktualnie obsługiwanych zbiorów danych:
    - **A2D2** - wewnątrz `./data/a2d2` powinien znaleźć się folder o nazwie `camera_lidar_semantic`.
    - **Cityscapes** - wewnątrz `./data/cityscapes` powinny minimalnie znaleźć się foldery o nazwie `gtFine` oraz `leftImg8bit`. Można również wczytać inne zbiory Cityscapes przeznaczone do segmentacji.

- Dlaczego należy rozpakować zbiór z pomocą np. *7zip*?

    *Eksplorator Plików* systemu Windows nie jest zoptymalizowany pod kątem archiwów `.tar`. Rozpakowywanie zbioru mogłoby zająć bardzo dużo czasu niezależnie od sprzętu.