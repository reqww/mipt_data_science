# Задание 1. Первичный анализ данных c Pandas

# В этом заданиии мы будем использовать данные SOCR по росту и весу 25 тысяч подростков.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize

# Считаем данные по росту и весу (weights_heights.csv, приложенный в задании) в объект Pandas DataFrame:
data = pd.read_csv("data.csv", index_col="Index")

# [2]. Посмотрите на первые 5 записей с помощью метода head Pandas DataFrame.
# Нарисуйте гистограмму распределения веса с помощью метода plot Pandas DataFrame.
# Сделайте гистограмму зеленой, подпишите картинку.

# Выводим часть данных на печать
print(data.head())

# Строим гистограмму
data.plot(
    y="Height",
    kind="hist",
    color="green",
    title="Распределение роста в дюймах",
)
plt.show()


# Один из эффективных методов первичного анализа данных - отображение попарных зависимостей признаков.
# Создается m \times mm×m графиков (m - число признаков), где по диагонали рисуются гистограммы распределения признаков,
# а вне диагонали - scatter plots зависимости двух признаков. Это можно делать с помощью метода scatter\_matrixscatter_matrix
# Pandas Data Frame или pairplot библиотеки Seaborn.

# Чтобы проиллюстрировать этот метод, интересней добавить третий признак.
# Создадим признак Индекс массы тела (BMI).
# Для этого воспользуемся удобной связкой метода apply Pandas DataFrame и lambda-функций Python.


def make_bmi(height_inch, weight_pound):
    METER_TO_INCH, KILO_TO_POUND = 39.37, 2.20462
    return (weight_pound / KILO_TO_POUND) / (height_inch / METER_TO_INCH) ** 2


data["BMI"] = data.apply(lambda row: make_bmi(row["Height"], row["Weight"]), axis=1)


# Строим график попарных зависимостей признаков при момощи scatter_matrix
scatter = pd.plotting.scatter_matrix(data)
plt.show()

# [3]. Постройте картинку, на которой будут отображены попарные зависимости признаков ,
# 'Height', 'Weight' и 'BMI' друг от друга. Используйте метод pairplot библиотеки Seaborn.

# Строим график попарных зависимостей признаков при момощи seaborn.pairplot
sns.set_theme()
sns.pairplot(data=data)
plt.show()


# [4]. Создайте в DataFrame data новый признак weight_category, который будет иметь 3 значения: 1 – если вес меньше 120 фунтов.
# (~ 54 кг.), 3 - если вес больше или равен 150 фунтов (~68 кг.), 2 – в остальных случаях.
# Постройте «ящик с усами» (boxplot), демонстрирующий зависимость роста от весовой категории.
# Используйте метод boxplot библиотеки Seaborn и метод apply Pandas DataFrame.
# Подпишите ось y меткой «Рост», ось x – меткой «Весовая категория».


def weight_category(weight):
    if weight < 120:
        return 1
    elif weight > 150:
        return 3

    return 2


data["weight_category"] = data.apply(lambda row: weight_category(row["Weight"]), axis=1)

ax = sns.boxplot(x="weight_category", y="Height", data=data)
ax.set(ylabel="Рост", xlabel="Весовая категория")
plt.show()

# [5]. Постройте scatter plot зависимости роста от веса,
# используя метод plot для Pandas DataFrame с аргументом kind='scatter'. Подпишите картинку.
data.plot(x="Weight", y="Height", kind="scatter", title="Зависимость роста от веса")
plt.show()


# Задание 2. Минимизация квадратичной ошибки


# [6]. Напишите функцию, которая по двум параметрам w0 и w1 вычисляет квадратичную ошибку приближения
# зависимости роста y от веса x прямой линией y = w0 + w1 * x


def error(w0, w1):
    s = 0
    for _, row in data.iterrows():
        s += (row["Height"] - (w0 + w1 * row["Weight"])) ** 2

    return s


# [7]. Проведите на графике из п. 5 Задания 1 две прямые, соответствующие значениям параметров (w0, w1) = (60, 0.05)
# и (w0, w1) = (50, 0.16)
# Используйте метод plot из matplotlib.pyplot, а также метод linspace библиотеки NumPy. Подпишите оси и график.


def plot_chart(w0, w1):
    def f(w0, w1, x):
        return w0 + w1 * x

    x = np.linspace(70, 170, 1000)

    data.plot(
        x="Weight",
        y="Height",
        kind="scatter",
        label="Зависимость роста от веса",
    )

    plt.plot(
        x,
        f(w0, w1, x),
        color="red",
        label=f"Зависимость роста от веса w0 = {w0}, w1 = {w1}",
    )

    plt.xlabel("Вес")
    plt.ylabel("Рост")

    plt.title(f"Зависимость роста от веса w0 = {w0}, w1 = {w1}")

    plt.legend()
    plt.show()


plot_chart(60, 0.05)
plot_chart(50, 0.16)

# [8] график зависимости функции ошибки

w0 = 50

w1 = np.linspace(0, 1, 100)
ax = plt.plot(w1, error(w0, w1))
plt.xlabel("W1")
plt.ylabel("Ошибка")
plt.title("График ошибки")
plt.show()


# Поиск оптимального веса w1
res = optimize.minimize_scalar(lambda w1: error(w0, w1), bounds=(-5, 5))
opt_w1 = round(res.x, 2)

plot_chart(w0, opt_w1)


# [10]. Постройте 3D-график зависимости функции ошибки, посчитанной в п.6
fig = plt.figure()
ax = fig.gca(projection="3d")  # get current axis

# Создаем массивы NumPy с координатами точек по осям X и У.
# Используем метод meshgrid, при котором по векторам координат
# создается матрица координат. Задаем нужную функцию Z(x, y).
w0 = np.arange(-5, 5, 0.25)
w1 = np.arange(-5, 5, 0.25)
w0, w1 = np.meshgrid(w0, w1)
Z = error(w0, w1)

# Наконец, используем метод *plot_surface* объекта
# типа Axes3DSubplot. Также подписываем оси.
surf = ax.plot_surface(w0, w1, Z)
ax.set_xlabel("Intercept")
ax.set_ylabel("Slope")
ax.set_zlabel("Error")
plt.show()


# [11]. С помощью метода minimize из scipy.optimize найдите минимум функции, определенной в п. 6
res = optimize.minimize(lambda w: error(w[0], w[1]), np.array([0, 0]), bounds=((-100, 100), (-5, 5)), method="L-BFGS-B")
w0, w1 = res.x
plot_chart(w0, w1)
