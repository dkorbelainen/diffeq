import numpy as np
import sympy as sp
from sympy import symbols, Matrix, exp, diff, simplify, expand

# Определяем символьную переменную
x = symbols('x')

# Матрица A
A = Matrix([
    [4, 2, -2],
    [-27, -9, 11],
    [0, 1, -1]
])

print("Матрица A:")
print(A)
print()

# Фундаментальная матрица Φ(x) из вашего решения
Phi = exp(-2 * x) * Matrix([
    [2, 6 * x - 2, 18 * x ** 2 - 12 * x - 2],
    [-3, 9 - 9 * x, 54 * x - 27 * x ** 2],
    [3, 9 * x, 27 * x ** 2]
])

print("Фундаментальная матрица Φ(x):")
print(Phi)
print()

# Вычисляем производную Φ'(x)
Phi_prime = diff(Phi, x)

print("Производная Φ'(x):")
print(Phi_prime)
print()

# Вычисляем AΦ(x)
A_Phi = A * Phi

print("Произведение AΦ(x):")
print(A_Phi)
print()

# Проверяем равенство Φ'(x) = AΦ(x)
difference = simplify(Phi_prime - A_Phi)

print("Разность Φ'(x) - AΦ(x):")
print(difference)
print()

# Проверяем, является ли разность нулевой матрицей
is_zero = all(simplify(element) == 0 for element in difference)

print("Является ли разность нулевой матрицей?", is_zero)
print()

# Дополнительная проверка: вычисляем определитель Φ(x)
det_Phi = Phi.det()
print("Определитель Φ(x):")
print(simplify(det_Phi))
print()

# Проверяем, что определитель не равен нулю (матрица невырожденная)
det_simplified = simplify(det_Phi)
print("Упрощенный определитель:")
print(det_simplified)
print()

# Проверим начальное условие Φ(0)
Phi_0 = Phi.subs(x, 0)
print("Φ(0):")
print(Phi_0)
print()

# Проверим, что собственные числа матрицы A действительно равны -2
eigenvals = A.eigenvals()
print("Собственные числа матрицы A:")
print(eigenvals)
print()

# Проверим каждый столбец фундаментальной матрицы отдельно
print("Проверка каждого столбца отдельно:")
for i in range(3):
    col = Phi.col(i)
    col_prime = diff(col, x)
    A_col = A * col
    col_diff = simplify(col_prime - A_col)

    print(f"Столбец {i + 1}:")
    print(f"y{i + 1} = {col}")
    print(f"y{i + 1}' = {col_prime}")
    print(f"Ay{i + 1} = {A_col}")
    print(f"Разность: {col_diff}")
    print(f"Равенство выполняется: {all(simplify(element) == 0 for element in col_diff)}")
    print()