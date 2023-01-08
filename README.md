# Simple object detector test bench

Небольшой проект для сравнения пропускной способности на CPU двух моделей детекции:
* [person-detection-0200](./models/openvino_model) - для запуска используется OpenVINO Runtime;
* [yolov4-tiny](./models/dnn) - для запуска используется OpenCV DNN.

Каждая модель наследуется от абстрактного класса AbstractTimedDetector, который предполагает наличие
методов `preprocess()`, `inference()`, `postprocess()`. Это позволяет проводить отдельные замеры каждого этапа
обработки изображения нейронной сетью.

### Зависимости

* open_model_zoo

Установка осуществляется следующим образом:
```shell script
git clone https://github.com/openvinotoolkit/open_model_zoo.git
cd open_model_zoo
pip3 install demos/common/python
```
Я использовал вот этот [коммит](https://github.com/openvinotoolkit/open_model_zoo/commit/2a5292c9d09fdb760559e1d8bcaf806bcf70cc07)

* git lfs

* Все остальные зависимости приведены в [requirements.txt](./requirements.txt)
```shell script
pip3 install -r /path/to/requirements.txt
```

Также стоит отметить, что для разработки использовался python3.8

### Использование

В проекте есть 2 основных скрипта [`demo.py`](./demo.py) и [`time_perfomance.py`](./time_perfomance.py).

* [`demo.py`](./demo.py) предназначен для покадровой визуализации работы детекторов: отрисовка боксов,
 уверенности модели и меток класса. Есть возможность получить время обработки каждого изображения, а также
 получить общую статистику (среднее, медиану, стандартное отклонение).
 
 Скрипт поддерживает интерфейс командной строки:
 - [__`input`__](./demo.py#L50) - Путь к тому, что будет подаваться в сеть. Это должно быть либо изображение,
  либо видео, либо папка с изображениями.
- [__`model`__](./demo.py#L53) - Тип модели. Допустимые значения: `openvino`, `dnn`.
- [__`output_dir`__](./demo.py#L55) - Папка, в которую будут сохраняться результаты работы детектора.
- [__`print_time`__](./demo.py#L57) - Надо ли выводить в консоль время обработки каждого изображения.

* [`time_perfomance.py`](./time_perfomance.py) простой скрипт, чтобы померить время работы сети на одной картинке
заданное число раз.
- [__`image`__](./demo.py#L50) - Путь к изображению, которое будет подаваться в сеть.
- [__`model`__](./demo.py#L53) - Тип модели. Допустимые значения: `openvino`, `dnn`.
- [__`num_iters`__](./demo.py#L55) - Количество прогонов изображения через сеть.
- [__`print_time`__](./demo.py#L57) - Надо ли выводить в консоль время обработки каждого изображения.
 
