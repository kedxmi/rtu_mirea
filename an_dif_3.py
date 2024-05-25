import random
import numpy as np
import matplotlib.pyplot as plt

# Алгоритм Кнута-Морриса-Пратта с подсчётом элементарных операций
def compute_lps(pattern, m, lps, ftr):
    length = 0
    lps[0] = 0
    i = 1
    ftr[0] += 2  # инициализация length и i

    while i < m:
        ftr[0] += 1  # проверка условия цикла
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
            ftr[0] += 3  # присваивание length, lps[i], и i
        else:
            if length != 0:
                length = lps[length - 1]
                ftr[0] += 1  # присваивание length
            else:
                lps[i] = 0
                i += 1
                ftr[0] += 2  # присваивание lps[i] и i

def KMP_search(pattern, text, ftr):
    m = len(pattern)
    n = len(text)

    lps = [0] * m
    j = 0  # индекс для pattern[]
    ftr[0] += 2  # инициализация m и n

    compute_lps(pattern, m, lps, ftr)

    i = 0  # индекс для text[]
    ftr[0] += 1  # инициализация i
    while i < n:
        ftr[0] += 1  # проверка условия цикла
        if pattern[j] == text[i]:
            i += 1
            j += 1
            ftr[0] += 2  # присваивание i и j

        if j == m:
            j = lps[j - 1]
            ftr[0] += 1  # присваивание j

        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
                ftr[0] += 1  # присваивание j
            else:
                i += 1
                ftr[0] += 1  # присваивание i

# Текст для поиска
text = """We are the champions
I've paid my dues
Time after time
I've done my sentence
But committed no crime
And bad mistakes
I've made a few
I've had my share of sand
Kicked in my face
But I've come through
And I need to go on and on and on and on

We are the champions - my friend
And we'll keep on fighting till the end
We are the champions
We are the champions
No time for losers
Cause we are the champions of the world

I've taken my bows
And my curtain calls
You've brought me fame and fortune
And everything that goes with it
I thank you all
But it's been no bed of roses no pleasure cruise
I consider it a challenge before the whole human race
And I ain't gonna lose
And I need to go on and on and on and on

We are the champions - my friend
And we'll keep on fighting till the end
We are the champions
We are the champions
No time for losers
‘Cause we are the champions of the world

We are the champions - my friend
And we'll keep on fighting till the end
We are the champions
We are the champions
No time for losers
Cause we are the champions"""

# 4. Запуск серии экспериментов
num_experiments = 20000
A = np.zeros(num_experiments)
text_len = len(text)

for exp in range(num_experiments):
    start_idx = random.randint(0, text_len - 3)
    pattern = text[start_idx:start_idx + 3]
    pattern += ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3))
    ftr = [0]  # счётчик элементарных операций
    KMP_search(pattern, text, ftr)
    A[exp] = ftr[0]

# 5. Использование функций библиотеки NumPy (Python)
# 5.1. Найти минимальное и максимальное значения массива A[ ].
min_val = np.min(A)
max_val = np.max(A)

print(f"Минимальное значение: {min_val}")
print(f"Максимальное значение: {max_val}")

# 5.2. Для массива A[ ] найти размерность и количество размерностей.
shape = A.shape
ndim = A.ndim

print(f"Размерность массива A: {shape}")
print(f"Количество размерностей массива A: {ndim}")

# 5.3. Отсортировать массив A[ ] по возрастанию значений элементов.
A_sorted = np.sort(A)

print(f"Отсортированный массив A: {A_sorted}")

# 5.4. Из отсортированного массива A[ ] вырезать массив A_50[ ], состоящий из всех строк и первых 50 элементов.
A_50 = A_sorted[:50]

print(f"Первые 50 элементов отсортированного массива A_50: {A_50}")

# 5.5. Из массива A_50[ ] извлечь массив B1[ ], состоящий из элементов, находящихся на нечётных местах и массив B2[ ], состоящий из элементов, находящихся на чётных местах.
B1 = A_50[::2]
B2 = A_50[1::2]

print(f"Массив B1 (нечётные элементы A_50): {B1}")
print(f"Массив B2 (чётные элементы A_50): {B2}")

# 5.6. Склеить массивы B1[ ] и B2[ ] вдоль вертикали.
B = np.vstack((B1, B2))

print(f"Склеенный массив B вдоль вертикали:\n{B}")

# 5.7. Из массива A_50[ ] извлечь массив C[ ], состоящий из чётных элементов.
C = A_50[A_50 % 2 == 0]

print(f"Массив C (чётные элементы A_50): {C}")

# 5.8. Найти статистические показатели исходного массива A[ ]: среднее значение, медиану, дисперсию, среднее квадратическое отклонение.
mean_val = np.mean(A)
median_val = np.median(A)
variance_val = np.var(A)
std_dev_val = np.std(A)

print(f"Среднее значение: {mean_val}")
print(f"Медиана: {median_val}")
print(f"Дисперсия: {variance_val}")
print(f"Среднее квадратическое отклонение: {std_dev_val}")

# 5.9. Построить гистограммы распределения значений A[ ] для 10, 20, 30, 40 и 50 интервалов равной длины.
intervals = [10, 20, 30, 40, 50]
for interval in intervals:
    plt.hist(A, bins=interval)
    plt.title(f'Гистограмма с интервалом {interval}')
    plt.xlabel('Значения')
    plt.ylabel('Частота')
    plt.show()

# 5.10. Построить квантили уровней 0.1, 0.3, 0.5, 0.9 для значений массива A[ ].
quantiles = np.quantile(A, [0.1, 0.3, 0.5, 0.9])

print(f"Квантили уровней 0.1, 0.3, 0.5, 0.9: {quantiles}")

# Построение графика квантилей
plt.plot([0.1, 0.3, 0.5, 0.9], quantiles, marker='o')
plt.title('Квантили A[]')
plt.xlabel('Уровень квантиля')
plt.ylabel('Значения')
plt.grid(True)
plt.show()

# 5.11. Построить ящик с усами для набора данных A[ ].
plt.boxplot(A)
plt.title('Ящик с усами A[]')
plt.xlabel('A[]')
plt.ylabel('Значения')
plt.show()
