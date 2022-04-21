This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
______________________________________

processing-of-sound-signals-by-a-NN - Нейросеть на Python, которая определяет звуки в категории.
За основу был взят код с этого сайта: https://nuancesprog.ru/p/6713/
______________________________________
Как использовать программу
1. Если нет набора данных dataset.csv, создать его с помощью модуля TT. Он использует звуковые файлы WAV в папке sounds, в которой дорожки расположены по категориям (этих папок нет в репозитории, так как есть файл dataset.csv);
2. Запустить модуль core, в котором создается экземпляр нейросети, проводится обучение и ее тестирование, после чего она сохраняется в model.h5;
3. Для распознавания звуков из папки sounds запустить модуль reshenie (и в дальнейшем использовать только этот модуль и два дополнительных train_test и util).
Результаты распознавания выводятся в консоли.
