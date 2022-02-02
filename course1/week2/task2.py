# Интерполировать функцию
# Рассмотрим сложную математическую функцию на отрезке [1, 15]:

# f(x) = sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)


# Сформируйте систему линейных уравнений (то есть задайте матрицу коэффициентов A и свободный вектор b) для
# многочлена первой степени, который должен совпадать с функцией f в точках 1 и 15. Решите данную систему с помощью
# функции scipy.linalg.solve. Нарисуйте функцию f и полученный многочлен. Хорошо ли он приближает исходную функцию?
#
# Повторите те же шаги для многочлена второй степени, который совпадает с функцией f в точках 1, 8 и 15. Улучшилось
# ли качество аппроксимации?
#
# Повторите те же шаги для многочлена третьей степени, который совпадает с функцией f в точках 1, 4, 10 и 15. Хорошо
# ли он аппроксимирует функцию? Коэффициенты данного многочлена (четыре числа в следующем порядке: w_0, w_1, w_2,
# w_3) являются ответом на задачу. Округлять коэффициенты не обязательно, но при желании можете произвести округление
# до второго знака (т.е. до числа вида 0.42)
#
# Запишите полученные числа в файл, разделив пробелами. Обратите внимание, что файл должен состоять из одной строки,
# в конце которой не должно быть переноса. Пример файла с решением вы можете найти в конце задания (submission-2.txt).

import math

import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt


def f(x):
    return math.sin(x/5) * math.exp(x/10) + 5 * math.exp(-x/2)


def task(*arr):
    A = [[x**i for i in range(len(arr))] for x in arr]

    b = [f(x) for x in arr]

    result = linalg.solve(A, b)

    def res_f(el):
        return sum([result[i]*el**i for i in range(len(result))])

    x = np.arange(min(arr), max(arr), 0.1)

    plt.plot(x, [f(x[i]) for i in range(len(x))])
    plt.plot(x, [res_f(x[i]) for i in range(len(x))])
    plt.show()

    print(f"w[i] для {', '.join(map(str, arr))}:", *list(map(lambda x: round(x, 2), result)))


task(1, 15)
task(1, 8, 15)
task(1, 4, 10, 15)
