# Линейная регрессия и стохастический градиентный спуск
# Задание основано на материалах лекций по линейной регрессии и градиентному спуску.
# Вы будете прогнозировать выручку компании в зависимости от уровня ее инвестиций в рекламу по TV, в газетах и по радио.

# Вы научитесь:
# решать задачу восстановления линейной регрессии
# реализовывать стохастический градиентный спуск для ее настройки
# решать задачу линейной регрессии аналитически


# 1. Загрузите данные из файла advertising.csv в объект pandas DataFrame. Источник данных.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_zero_arr(n):
    return np.ones(n).reshape((n, 1))


def mserror(y, y_pred):
    return ((y - y_pred) ** 2).mean(axis=0)


def normal_equation(x, y):
    return np.dot(np.linalg.pinv(x), y)


def linear_prediction(x, w):
    return np.dot(x, w)


def stochastic_gradient_step(x, y, w, train_ind, eta=0.01):
    arr = linear_prediction(x[train_ind], w) - y[train_ind]
    l = len(x)

    grad = [el * arr for el in x[train_ind]]

    return w - 2 * eta / l * np.array(grad)


def stochastic_gradient_descent(x, y, w_init, eta=1e-2, max_iter=1e4, min_weight_dist=1e-8, seed=42, verbose=False):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом.
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    # Будем порождать псевдослучайные числа
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)

    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # порождаем псевдослучайный
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(x.shape[0])

        new_weights = stochastic_gradient_step(x, y, w, random_ind)

        np.linalg.norm(w - new_weights)

        w = new_weights

        iter_num += 1

        y_pred = linear_prediction(x, w)

        errors.append(mserror(y, y_pred))

    return w, errors


if __name__ == "__main__":
    data = pd.read_csv("advertising.csv")

    print(data.head())

    x = data[["TV", "Radio", "Newspaper"]].to_numpy()
    y = data["Sales"].to_numpy()

    means, stds = x.mean(axis=0), x.std(axis=0)

    x = (x - means) / stds

    x = np.hstack((x, get_zero_arr(len(x))))

    # Какова среднеквадратичная ошибка прогноза значений Sales, если всегда предсказывать медианное значение
    # Sales по исходной выборке? Полученный результат, округленный до 3 знаков после запятой, является ответом на '1 задание'.

    print(f"answer 1 = {round(mserror(y, np.median(y)), 3)}")

    weights = normal_equation(x, y)

    print(f"analytics weights = {weights}")

    # Какие продажи предсказываются линейной моделью с весами, найденными с помощью нормального уравнения,
    # в случае средних инвестиций в рекламу по ТВ, радио и в газетах? (то есть при нулевых значениях масштабированных признаков TV, Radio и Newspaper).
    # Полученный результат, округленный до 3 знаков после запятой, является ответом на '2 задание'.

    mean_normed_x = np.hstack((np.zeros((1, 3)), get_zero_arr(1)))

    y_pred = linear_prediction(mean_normed_x, weights)

    print(f"answer 2 = {round(y_pred[0], 3)}")

    # 4. Напишите функцию linear_prediction, которая принимает на вход матрицу X
    # и вектор весов линейной модели w, а возвращает вектор прогнозов в виде линейной
    # комбинации столбцов матрицы X с весами w.

    # Какова среднеквадратичная ошибка прогноза значений Sales в виде линейной модели с весами,
    # найденными с помощью нормального уравнения? Полученный результат, округленный
    # до 3 знаков после запятой, является ответом на '3 задание'

    y_pred = linear_prediction(x, weights)

    print(f"answer 3 = {round(mserror(y, y_pred), 3)}")

    # 5. Напишите функцию stochastic_gradient_step, реализующую шаг стохастического градиентного спуска для линейной регрессии.
    # Функция должна принимать матрицу X, вектора y и w, число train_ind - индекс объекта обучающей выборки (строки матрицы X),
    # по которому считается изменение весов, а также число \etaη (eta) - шаг градиентного спуска (по умолчанию eta=0.01).
    # Результатом будет вектор обновленных весов. Наша реализация функции будет явно написана для данных с 3 признаками,
    # но несложно модифицировать для любого числа признаков, можете это сделать.

    # Запустите 10^5 итераций стохастического градиентного спуска.
    # Укажите вектор начальных весов w_init, состоящий из нулей.
    # Оставьте параметры eta и seed равными их значениям по умолчанию (eta=0.01, seed=42 - это важно для проверки ответов).

    stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(x, y, np.zeros(4), max_iter=1e5)

    print(f"grad desc weights = {stoch_grad_desc_weights}")

    plt.plot(range(len(stoch_errors_by_iter[:50])), stoch_errors_by_iter[:50])
    plt.xlabel("Iteration number")
    plt.ylabel("MSE")
    plt.show()

    print(f"answer 4 = {round(stoch_errors_by_iter[-1], 3)}")
