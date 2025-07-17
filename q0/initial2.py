import sympy as sp
import numpy as np
from sympy import symbols, Matrix, exp, log, integrate, simplify, factor

# Определяем символы
x, alpha0, beta0, gamma0 = symbols('x alpha_0 beta_0 gamma_0')

print("Решение системы Φ(x)c'(x) = q^(0)(x)")
print("=" * 50)

# Фундаментальная матрица Φ(x)
Phi = Matrix([
    [2, 6 * x - 2, 18 * x ** 2 - 12 * x - 2],
    [-3, 9 - 9 * x, 54 * x - 27 * x ** 2],
    [3, 9 * x, 27 * x ** 2]
])

print("Фундаментальная матрица Φ(x) =")
print(Phi)
print()

# Правая часть q^(0)(x) = x^(-2) * e^(-2x) * [α₀, β₀, γ₀]
# Но поскольку Φ(x) содержит множитель e^(-2x), система для c'(x) имеет вид:
# Φ(x) * c'(x) = q^(0)(x)
# где q^(0)(x) = x^(-2) * e^(-2x) * [α₀, β₀, γ₀]

# Правая часть системы для c'(x) (без множителя e^(-2x))
q0_reduced = Matrix([alpha0 * x ** (-2), beta0 * x ** (-2), gamma0 * x ** (-2)])

print("Правая часть системы для c'(x):")
print("q^(0)_reduced =", q0_reduced)
print()

# Решаем систему Φ(x) * c'(x) = q^(0)_reduced
print("Решаем систему методом Крамера:")
print("Φ(x) * c'(x) = q^(0)_reduced")
print()

# Вычисляем определитель Φ(x)
det_Phi = Phi.det()
print("det(Φ) =", simplify(det_Phi))
print()

# Проверяем, что определитель не равен нулю
if det_Phi != 0:
    print("Определитель не равен нулю, система имеет единственное решение")
    print()

    # Решаем систему
    c_prime = Phi.inv() * q0_reduced

    print("Решение c'(x) = Φ^(-1)(x) * q^(0)_reduced:")
    print()

    for i in range(3):
        c_prime_i = simplify(c_prime[i])
        print(f"c_{i + 1}'(x) = {c_prime_i}")
        print()

        # Дополнительно упростим и разложим на множители
        factored = factor(c_prime_i)
        print(f"c_{i + 1}'(x) (факторизованный) = {factored}")
        print()

    print("=" * 50)
    print("Интегрирование для нахождения c(x):")
    print()

    # Интегрируем c'(x) для получения c(x)
    c_solutions = []
    for i in range(3):
        print(f"Интегрируем c_{i + 1}'(x):")
        c_integrated = integrate(c_prime[i], x)
        c_simplified = simplify(c_integrated)
        c_solutions.append(c_simplified)
        print(f"c_{i + 1}(x) = {c_simplified}")
        print()

    print("=" * 50)
    print("Проверка решения:")
    print()

    # Проверяем решение, подставляя обратно в систему
    for i in range(3):
        # Берем производную от найденного c_i(x)
        c_derivative = sp.diff(c_solutions[i], x)
        print(f"Проверка: d/dx[c_{i + 1}(x)] = {simplify(c_derivative)}")

        # Сравниваем с исходным c'_i(x)
        difference = simplify(c_derivative - c_prime[i])
        print(f"Разность с c_{i + 1}'(x) = {difference}")
        print()

    print("=" * 50)
    print("Окончательные выражения для c'(x):")
    print()

    # Выводим окончательные выражения в удобном виде
    for i in range(3):
        print(f"c_{i + 1}'(x) = {simplify(c_prime[i])}")

        # Приведем к общему знаменателю и упростим
        numerator = sp.numer(c_prime[i])
        denominator = sp.denom(c_prime[i])
        print(f"       = ({numerator}) / ({denominator})")
        print()

else:
    print("Ошибка: определитель равен нулю!")

# Дополнительно: проверим правильность фундаментальной матрицы
print("=" * 50)
print("Проверка фундаментальной матрицы:")
print()

# Матрица A
A = Matrix([
    [4, 2, -2],
    [-27, -9, 11],
    [0, 1, -1]
])

print("Матрица A =")
print(A)
print()

# Полная фундаментальная матрица с множителем e^(-2x)
Phi_full = exp(-2 * x) * Phi

print("Полная фундаментальная матрица Φ(x) = e^(-2x) * матрица =")
print("e^(-2x) *")
print(Phi)
print()

# Проверяем, что Φ'(x) = A * Φ(x)
Phi_full_prime = sp.diff(Phi_full, x)
A_Phi_full = A * Phi_full

print("Φ'(x) (полная производная) =")
print(simplify(Phi_full_prime))
print()

print("A * Φ(x) (полная) =")
print(simplify(A_Phi_full))
print()

difference_matrix = simplify(Phi_full_prime - A_Phi_full)
print("Разность Φ'(x) - A*Φ(x) =")
print(difference_matrix)

if difference_matrix == Matrix.zeros(3, 3):
    print("✓ Фундаментальная матрица корректна!")
else:
    print("✗ Ошибка в фундаментальной матрице!")