# Segmentacja semantyczna w systemach nowoczesnych pojazdów

Niniejsze repozytorium zawiera projekt magisterski realizowany w ramach pracy pt. **Zastosowanie sztucznych sieci neuronowych w systemach pojazdów autonomicznych** autorstwa **Patryka Piróg**, realizowanej w ramach grupy badawczej **ZSDiiZ** (Uniwersytet Śląski w Katowicach).

Projekt skupia się wokół zastosowania technik uczenia głębokiego w treningu modeli realizujących zadanie *segmentacji semantycznej*. Głównym celem projektu jest przygotowanie skryptów w języku *Python*, które posłużą do **treningu**, **ewaluacji** oraz **inferencji** modeli konwolucyjnych w celu porównania ich wydajności.

## Porównywane architektury
- [**resnet50**](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) ze wstępnie wytrenowanymi wagami.

## Testowane zbiory danych
- [**A2D2**](https://www.a2d2.audi/en/) - zbiór danych grupy VAG, składający się z wielu scen uchwyconych w niemieckich miastach. W skład zbioru wchodzą obrazy `.png` z wielu kamer w rozdzielczości *1920x1208*, dane z *magistrali CAN* oraz *LIDARu*.

## Instrukcja uruchomienia
1. Zainstalować zależności ``pip install -r requirements.txt``.
2. Pobrać ręcznie [pełny zbiór danych A2D2](https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic.tar), rozpakować go z użyciem np. [7zip](https://www.7-zip.org/download.html), umieścić go w katalogu ``./data/``.

### FAQ przy instalacji
- Czy lista `requirements.txt` jest kompletna?

    Lista może nie być kompletna. Zalecane jest ręczne doinstalowanie brakujących zależności.

- Co właściwie powinno znajdować się w katalogu  `./data/`?

    W katalogu `./data/` powinien znaleść się folder `camera_lidar_semantic`,  który zawiera pełną strukturę zbioru danych [A2D2](https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic.tar).

- Dlaczego należy rozpakować zbiór z pomocą np. *7zip*?

    *Eksplorator Plików* systemu Windows nie jest zoptymalizowany pod kątem archiwów `.tar`. Rozpakowywanie zbioru mogłoby zająć bardzo dużo czasu niezależnie od sprzętu.