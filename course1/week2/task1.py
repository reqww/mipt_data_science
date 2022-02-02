# coding=utf-8

import numpy as np
import re
import collections
from scipy import spatial

# Выполните следующие шаги:
# Скачайте файл с предложениями (sentences.txt).

# Каждая строка в файле соответствует одному предложению.
# Считайте их, приведите каждую к нижнему регистру с помощью строковой функции lower().

# Произведите токенизацию, то есть разбиение текстов на слова.
# Для этого можно воспользоваться регулярным выражением, которое считает разделителем любой символ,
# не являющийся буквой: re.split('[^a-z]', t). Не забудьте удалить пустые слова после разделения.

# Составьте список всех слов, встречающихся в предложениях.
# Сопоставьте каждому слову индекс от нуля до (d - 1), где d — число различных слов в предложениях.
# Для этого удобно воспользоваться структурой dict.

# Создайте матрицу размера n * d, где n — число предложений.
# Заполните ее: элемент с индексом (i, j) в этой матрице должен быть равен количеству
# вхождений j-го слова в i-е предложение. У вас должна получиться матрица размера 22 * 254.

# Найдите косинусное расстояние от предложения в самой первой строке (In comparison to dogs, cats have not undergone...)
# до всех остальных с помощью функции scipy.spatial.distance.cosine.
# Какие номера у двух предложений, ближайших к нему по этому расстоянию (строки нумеруются с нуля)?
# Эти два числа и будут ответами на задание. Само предложение (In comparison to dogs, cats have not undergone... )
# имеет индекс 0.

# Запишите полученные числа в файл, разделив пробелом.
# Обратите внимание, что файл должен состоять из одной строки, в конце которой не должно быть переноса.
# Пример файла с решением вы можете найти в конце задания (submission-1.txt).

# Совпадают ли ближайшие два предложения по тематике с первым? Совпадают ли тематики у следующих по близости предложений
# Разумеется, использованный вами метод крайне простой.
# Например, он не учитывает формы слов (так, cat и cats он считает разными словами,
# хотя по сути они означают одно и то же), не удаляет из текстов артикли и прочие ненужные слова.
# Позже мы будем подробно изучать анализ текстов, где выясним, как достичь высокого качества
# в задаче поиска похожих предложений.


def get_word_dict():
    with open("task1.txt", "r") as f:
        all_lines = []
        def_dict = collections.defaultdict(lambda: -1)

        number = 0

        for line in f.readlines():
            line_arr = []

            for word in filter(lambda x: bool(x), re.split('[^a-z]', line.lower())):
                if def_dict[word] == -1:
                    def_dict[word] = len(def_dict) - 1

                line_arr.append(word)

            all_lines.append(line_arr)

            number += 1

    return def_dict, number, all_lines


def get_word_matrix(word_dict, number_sentences, all_lines):
    matrx = np.zeros((number_sentences, len(word_dict)))

    for i, line in enumerate(all_lines):
        for word in line:
            matrx[i][word_dict[word]] += 1

    return matrx


def get_last_two_cosine_distance(matrx):
    sent = matrx[0]
    arr = []

    for i, sentence in enumerate(matrx[1:]):
        arr.append((spatial.distance.cosine(sent, sentence), i+1))

    return list(sorted(arr, key=lambda x: x[0]))[:2]


d, n, lines = get_word_dict()


matr = get_word_matrix(d, n, lines)

print(get_last_two_cosine_distance(matr))

