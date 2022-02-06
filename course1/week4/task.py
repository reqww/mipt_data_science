# ДИСКЛЕЙМЕР

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import expon, norm

# Выберите ваше любимое непрерывное распределение (чем меньше оно будет похоже на нормальное, тем интереснее;
# попробуйте выбрать какое-нибудь распределение из тех, что мы не обсуждали в курсе). Сгенерируйте из него выборку
# объёма 1000, постройте гистограмму выборки и нарисуйте поверх неё теоретическую плотность распределения вашей
# случайной величины (чтобы величины были в одном масштабе, не забудьте выставить у гистограммы значение параметра
# normed=True).


def print_exp_pdf(lamb, a, b, n):
    # Строим экспоненциальное распределение
    exp_rv = expon(scale=1/lamb)

    x = np.linspace(a, b, (b-a)*8)

    # Генерируем значения из экспоненциального распределения
    vals = exp_rv.rvs(n)

    # Строим теоретическую функцию плотности распределения
    pdf = exp_rv.pdf(x)

    # Строим график
    plt.plot(x, pdf)
    plt.hist(vals, bins=x, density=True)

    plt.ylabel('$p(x)$')
    plt.xlabel('$x$')

    plt.show()


l = 2
left, right = 0, 3
size = 1000


print_exp_pdf(l, left, right, size)

# Ваша задача — оценить распределение выборочного среднего вашей случайной величины при разных объёмах выборок. Для
# этого при трёх и более значениях n (например, 5, 10, 50) сгенерируйте 1000 выборок объёма n и постройте гистограммы
# распределений их выборочных средних. Используя информацию о среднем и дисперсии исходного распределения (её можно
# без труда найти в википедии), посчитайте значения параметров нормальных распределений, которыми,
# согласно центральной предельной теореме, приближается распределение выборочных средних. Обратите внимание: для
# подсчёта значений этих параметров нужно использовать именно теоретические среднее и дисперсию вашей случайной
# величины, а не их выборочные оценки. Поверх каждой гистограммы нарисуйте плотность соответствующего нормального
# распределения (будьте внимательны с параметрами функции, она принимает на вход не дисперсию, а стандартное
# отклонение).
#
# Опишите разницу между полученными распределениями при различных значениях n. Как меняется точность аппроксимации
# распределения выборочных средних нормальным с ростом n?
#

n_arr = [5, 10, 50, 150]

n_sample = 1000

exp_rv = expon(scale=1/l)

x_arr = np.linspace(0, 4, 100)


def print_hist_and_norm(mu, scale, x, means, smaple_n):
    # Строим гистограмму по средним выборок
    plt.hist(means, label=f'hist with n = {smaple_n}', density=True)

    # Строим нормальное распределение с параметрами отклонения и среднего выборки средних
    new_dist = norm(loc=mu, scale=scale)

    # Строим график
    pdf = new_dist.pdf(x)
    plt.plot(x, pdf, linewidth=2.5)
    plt.legend()
    plt.xlabel('$\\bar{X}_n$')
    plt.ylabel('$f(\\bar{X}_n)$')
    plt.title(f"n = {smaple_n}")
    plt.show()


for n in n_arr:
    arr = []
    for _ in range(n_sample):
        arr.append(exp_rv.rvs(n).mean())

    # Высчитываем теоретические средние и отклонение
    # Среднее будет равняться среднему распределения
    # Среднеквадратичное отклонения высчитывает по формуле sqrt(Dx / n), где Dx - дисперсия х
    print_hist_and_norm(exp_rv.mean(), (exp_rv.var()/n)**0.5, x_arr, arr, n)


# Вывод
# Очевидно, что с ростом количества значений в выборках, ЦПТ начинает аппроксимировать распределение средних более точно
# Т.е. точность аппроксимации увеличивается