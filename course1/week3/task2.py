# Задача 2. Глобальная оптимизация
# Теперь попробуем применить к той же функции f(x) метод глобальной оптимизации — дифференциальную эволюцию.
#
# Изучите документацию и примеры использования функции scipy.optimize.differential_evolution.
#
# Обратите внимание, что границы значений аргументов функции представляют собой список кортежей (list,
# в который помещены объекты типа tuple). Даже если у вас функция одного аргумента, возьмите границы его значений в
# квадратные скобки, чтобы передавать в этом параметре список из одного кортежа, т.к. в реализации
# scipy.optimize.differential_evolution длина этого списка используется чтобы определить количество аргументов функции.
#
# Запустите поиск минимума функции f(x) с помощью дифференциальной эволюции на промежутке [1, 30]. Полученное
# значение функции в точке минимума - ответ в задаче 2. Запишите его с точностью до второго знака после запятой. В
# этой задаче ответ - только одно число.
#
# Заметьте, дифференциальная эволюция справилась с задачей поиска глобального минимума на отрезке, т.к. по своему
# устройству она предполагает борьбу с попаданием в локальные минимумы.
#
# Сравните количество итераций, потребовавшихся BFGS для нахождения минимума при хорошем начальном приближении,
# с количеством итераций, потребовавшихся дифференциальной эволюции. При повторных запусках дифференциальной эволюции
# количество итераций будет меняться, но в этом примере, скорее всего, оно всегда будет сравнимым с количеством
# итераций BFGS. Однако в дифференциальной эволюции за одну итерацию требуется выполнить гораздо больше действий,
# чем в BFGS. Например, можно обратить внимание на количество вычислений значения функции (nfev) и увидеть,
# что у BFGS оно значительно меньше. Кроме того, время работы дифференциальной эволюции очень быстро растет с
# увеличением числа аргументов функции.
#

from scipy import optimize
from math import sin, exp


dist = [1, 30]


def f(x):
    return sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)


def find_min(func, coord):
    return optimize.differential_evolution(func, [coord])


res = find_min(f, dist)
print(res)
print(round(f(res.x[0]), 2))
