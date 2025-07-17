import numpy as np
import sympy as sp
from sympy import symbols, Matrix, exp, diff, simplify, expand, log, integrate, Abs

# Определяем символьные переменные
x = symbols('x', real=True, positive=True)
alpha0, beta0, gamma0 = symbols('alpha0 beta0 gamma0', real=True)

print("ПРАВИЛЬНОЕ РЕШЕНИЕ ДЛЯ НЕОДНОРОДНОСТИ q^(0)")
print("=" * 60)

# Правильные выражения для c'(x) (из скрипта)
c1_prime = -9*alpha0/2 - beta0 - beta0/(3*x) + 2*gamma0 - gamma0/(3*x) + gamma0/(3*x**2)
c2_prime = (27*alpha0*x + beta0*(6*x + 1) - gamma0*(12*x - 1))/(9*x**2)
c3_prime = (-9*alpha0 - 2*beta0 + 4*gamma0)/(18*x**2)

print("ШАГ 1: Правильные выражения для c'(x)")
print(f"c₁'(x) = {c1_prime}")
print(f"c₂'(x) = {c2_prime}")
print(f"c₃'(x) = {c3_prime}")
print()

# Упростим c2_prime для удобства
c2_prime_simplified = simplify(c2_prime)
print(f"c₂'(x) упрощенное = {c2_prime_simplified}")
print()

print("ШАГ 2: Интегрирование c'(x)")
print("-" * 30)

# Интегрируем каждое выражение
print("Интегрирование c₁'(x):")
c1_terms = [-9*alpha0/2, -beta0, -beta0/(3*x), 2*gamma0, -gamma0/(3*x), gamma0/(3*x**2)]
print("Разбиваем на слагаемые:")
for i, term in enumerate(c1_terms):
    print(f"  Слагаемое {i+1}: {term}")

c1_integrated = integrate(c1_prime, x)
print(f"c₁(x) = {c1_integrated}")
print()

print("Интегрирование c₂'(x):")
c2_integrated = integrate(c2_prime, x)
print(f"c₂(x) = {c2_integrated}")
print()

print("Интегрирование c₃'(x):")
c3_integrated = integrate(c3_prime, x)
print(f"c₃(x) = {c3_integrated}")
print()

# Упростим полученные выражения
print("ШАГ 3: Упрощение интегрированных выражений")
print("-" * 45)

c1_simplified = simplify(c1_integrated)
c2_simplified = simplify(c2_integrated)
c3_simplified = simplify(c3_integrated)

print(f"c₁(x) = {c1_simplified}")
print(f"c₂(x) = {c2_simplified}")
print(f"c₃(x) = {c3_simplified}")
print()

# Запишем в более удобном виде
print("ШАГ 4: Запись в стандартном виде")
print("-" * 35)

# Для c1(x)
print("Анализ c₁(x):")
print("Константная часть:", -9*alpha0*x/2 - beta0*x + 2*gamma0*x)
print("Логарифмическая часть:", -beta0*log(x)/3 - gamma0*log(x)/3)
print("Степенная часть:", -gamma0/(3*x))
print()

# Для c2(x)
print("Анализ c₂(x):")
print("Разложим дробь в c₂'(x):")
c2_prime_expanded = expand(c2_prime)
print(f"c₂'(x) = {c2_prime_expanded}")

# Интегрируем по частям
c2_log_part = (27*alpha0 + 6*beta0 - 12*gamma0)*log(x)/9
c2_const_part = integrate(beta0/(9*x**2) + gamma0/(9*x**2), x)
c2_rational_part = integrate(-gamma0/(9*x), x)

print(f"Логарифмическая часть: {c2_log_part}")
print(f"Рациональная часть: {c2_const_part}")
print()

# Для c3(x)
print("Анализ c₃(x):")
print(f"c₃(x) = {c3_simplified}")
print()

# Теперь вычислим частное решение
print("ШАГ 5: Вычисление частного решения ψ^(0)(x)")
print("-" * 50)

# Фундаментальная матрица
Phi = exp(-2*x) * Matrix([
    [2, 6*x - 2, 18*x**2 - 12*x - 2],
    [-3, 9 - 9*x, 54*x - 27*x**2],
    [3, 9*x, 27*x**2]
])

# Вектор c(x)
c_vector = Matrix([c1_simplified, c2_simplified, c3_simplified])

# Частное решение
psi0 = Phi * c_vector

print("Частное решение ψ^(0)(x) = Φ(x) · c(x):")
for i in range(3):
    psi_i_simplified = simplify(psi0[i])
    print(f"ψ_{i+1}^(0)(x) = {psi_i_simplified}")
print()

# Проверка правильности
print("ШАГ 6: Проверка правильности решения")
print("-" * 40)

# Неоднородность
q0 = exp(-2*x) * Matrix([alpha0, beta0, gamma0]) / x**2

# Матрица системы
A = Matrix([
    [4, 2, -2],
    [-27, -9, 11],
    [0, 1, -1]
])

# Проверяем: ψ'^(0) - Aψ^(0) = q^(0)
psi0_prime = diff(psi0, x)
check = psi0_prime - A * psi0 - q0

print("Проверка: ψ'^(0) - Aψ^(0) - q^(0) =")
all_zero = True
for i in range(3):
    check_simplified = simplify(check[i])
    print(f"Компонента {i+1}: {check_simplified}")
    if check_simplified != 0:
        all_zero = False

print(f"\nРешение правильное: {all_zero}")

# Выведем окончательные выражения в развернутом виде
print("\n" + "=" * 60)
print("ОКОНЧАТЕЛЬНЫЕ ПРАВИЛЬНЫЕ ВЫРАЖЕНИЯ")
print("=" * 60)

for i in range(3):
    expanded_psi = expand(psi0[i])
    print(f"ψ_{i+1}^(0)(x) = {expanded_psi}")
    print()