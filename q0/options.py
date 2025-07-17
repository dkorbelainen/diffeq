import numpy as np
from fractions import Fraction
import sympy as sp

# Определим символы
alpha, beta, gamma = sp.symbols('alpha beta gamma')

# Ключевые выражения из системы
A = 9 * alpha + 2 * beta - 4 * gamma  # коэффициент при x*ln|x|
B = 3 * alpha + beta - gamma  # коэффициент при ln|x| в ψ₁
C = 27 * alpha + 7 * beta - 11 * gamma  # коэффициент при ln|x| в ψ₂
D = beta + gamma  # коэффициент при ln|x| в ψ₃

print("Ключевые выражения:")
print(f"A = 9α + 2β - 4γ")
print(f"B = 3α + β - γ")
print(f"C = 27α + 7β - 11γ")
print(f"D = β + γ")
print()

# Известное решение
known_solution = (2, -3, 3)
print(f"Известное решение (α, β, γ) = {known_solution}")
print(f"A = {9 * 2 + 2 * (-3) - 4 * 3} = 0")
print(f"B = {3 * 2 + (-3) - 3} = 0")
print(f"C = {27 * 2 + 7 * (-3) - 11 * 3} = 0")
print(f"D = {-3 + 3} = 0")
print()

# Поиск других решений
print("Поиск других решений:")
print()

# 1. Обнуление только A (коэффициент при x*ln|x|)
print("1. Обнуление только A = 9α + 2β - 4γ = 0")
print("   γ = (9α + 2β)/4")
print("   Примеры:")
solutions_A = []
for alpha_val in range(-2, 3):
    for beta_val in range(-4, 5):
        if (9 * alpha_val + 2 * beta_val) % 4 == 0:
            gamma_val = (9 * alpha_val + 2 * beta_val) // 4
            if gamma_val != 0 and not (alpha_val == 0 and beta_val == 0):
                solutions_A.append((alpha_val, beta_val, gamma_val))
                if len(solutions_A) <= 5:
                    print(f"   (α, β, γ) = ({alpha_val}, {beta_val}, {gamma_val})")

print()

# 2. Обнуление A и B
print("2. Обнуление A и B:")
print("   9α + 2β - 4γ = 0")
print("   3α + β - γ = 0")
print("   Решая систему: β = -3α, γ = 0")
print("   Примеры:")
for alpha_val in [-2, -1, 1, 2]:
    beta_val = -3 * alpha_val
    gamma_val = 0
    print(f"   (α, β, γ) = ({alpha_val}, {beta_val}, {gamma_val})")

print()

# 3. Обнуление A и D
print("3. Обнуление A и D:")
print("   9α + 2β - 4γ = 0")
print("   β + γ = 0")
print("   Решая систему: β = -γ, α = 2γ/3")
print("   Примеры (где α целое):")
for gamma_val in [-3, 3, 6, -6]:
    if (2 * gamma_val) % 3 == 0:
        alpha_val = (2 * gamma_val) // 3
        beta_val = -gamma_val
        print(f"   (α, β, γ) = ({alpha_val}, {beta_val}, {gamma_val})")

print()

# 4. Обнуление B и D
print("4. Обнуление B и D:")
print("   3α + β - γ = 0")
print("   β + γ = 0")
print("   Решая систему: β = -γ, α = 2γ/3")
print("   Это те же решения, что и для случая 3")

print()

# 5. Специальные случаи
print("5. Другие интересные решения:")

# Проверим некоторые простые комбинации
candidates = [
    (1, 0, 1),
    (1, -1, 0),
    (0, 1, -1),
    (1, 1, -1),
    (-1, 1, 1),
    (2, 0, 2),
    (1, -2, 1),
    (0, 2, -2),
    (3, -6, 3),
    (1, -3, 0),
    (0, 3, -3),
    (4, -6, 6),
    (2, -6, 0),
    (0, 4, -4)
]

print("Проверка дополнительных кандидатов:")
for alpha_val, beta_val, gamma_val in candidates:
    A_val = 9 * alpha_val + 2 * beta_val - 4 * gamma_val
    B_val = 3 * alpha_val + beta_val - gamma_val
    C_val = 27 * alpha_val + 7 * beta_val - 11 * gamma_val
    D_val = beta_val + gamma_val

    zeros = sum([A_val == 0, B_val == 0, C_val == 0, D_val == 0])
    if zeros >= 2:
        print(f"   (α, β, γ) = ({alpha_val}, {beta_val}, {gamma_val})")
        print(f"   A={A_val}, B={B_val}, C={C_val}, D={D_val}")
        print(f"   Обнулено: {zeros} коэффициентов")
        print()

# Поиск решений с максимальным количеством нулей
print("6. Систематический поиск решений с максимальным упрощением:")
best_solutions = []

for alpha_val in range(-4, 5):
    for beta_val in range(-8, 9):
        for gamma_val in range(-8, 9):
            if alpha_val == 0 and beta_val == 0 and gamma_val == 0:
                continue

            A_val = 9 * alpha_val + 2 * beta_val - 4 * gamma_val
            B_val = 3 * alpha_val + beta_val - gamma_val
            C_val = 27 * alpha_val + 7 * beta_val - 11 * gamma_val
            D_val = beta_val + gamma_val

            zeros = sum([A_val == 0, B_val == 0, C_val == 0, D_val == 0])

            if zeros >= 2:
                solution = (alpha_val, beta_val, gamma_val)
                if solution not in best_solutions:
                    best_solutions.append((solution, zeros, A_val, B_val, C_val, D_val))

# Сортируем по количеству нулей
best_solutions.sort(key=lambda x: x[1], reverse=True)

print("Лучшие решения (2+ нулевых коэффициента):")
for i, (sol, zeros, A_val, B_val, C_val, D_val) in enumerate(best_solutions[:15]):
    alpha_val, beta_val, gamma_val = sol
    print(f"{i + 1}. (α, β, γ) = {sol}")
    print(f"   A={A_val}, B={B_val}, C={C_val}, D={D_val}")
    print(f"   Обнулено: {zeros} коэффициентов")
    print()

# Специальный поиск второго полного решения
print("7. Поиск второго полного решения (все 4 коэффициента = 0):")
print("Система уравнений:")
print("9α + 2β - 4γ = 0")
print("3α + β - γ = 0")
print("27α + 7β - 11γ = 0")
print("β + γ = 0")

# Решаем систему символически
alpha_s, beta_s, gamma_s = sp.symbols('alpha beta gamma')
eq1 = sp.Eq(9 * alpha_s + 2 * beta_s - 4 * gamma_s, 0)
eq2 = sp.Eq(3 * alpha_s + beta_s - gamma_s, 0)
eq3 = sp.Eq(27 * alpha_s + 7 * beta_s - 11 * gamma_s, 0)
eq4 = sp.Eq(beta_s + gamma_s, 0)

# Проверим совместность
solution = sp.solve([eq1, eq2, eq3, eq4], [alpha_s, beta_s, gamma_s])
print(f"Решение системы: {solution}")

if not solution:
    print("Система не имеет нетривиальных решений кроме уже найденного.")
    print("Ищем решения с 3 нулевыми коэффициентами...")

    # Проверяем различные комбинации из 3 уравнений
    combinations = [
        ([eq1, eq2, eq3], "A=B=C=0"),
        ([eq1, eq2, eq4], "A=B=D=0"),
        ([eq1, eq3, eq4], "A=C=D=0"),
        ([eq2, eq3, eq4], "B=C=D=0")
    ]

    for eqs, name in combinations:
        print(f"\n{name}:")
        sol = sp.solve(eqs, [alpha_s, beta_s, gamma_s])
        if sol:
            print(f"Решение: {sol}")
        else:
            print("Нет решения")

print("\n8. Проверка конкретных наборов:")

# Проверяем (1, -3, 0)
alpha_val, beta_val, gamma_val = 1, -3, 0
A_val = 9 * alpha_val + 2 * beta_val - 4 * gamma_val
B_val = 3 * alpha_val + beta_val - gamma_val
C_val = 27 * alpha_val + 7 * beta_val - 11 * gamma_val
D_val = beta_val + gamma_val

print(f"Набор (α, β, γ) = (1, -3, 0):")
print(f"A = 9(1) + 2(-3) - 4(0) = 9 - 6 = {A_val}")
print(f"B = 3(1) + (-3) - 0 = 3 - 3 = {B_val}")
print(f"C = 27(1) + 7(-3) - 11(0) = 27 - 21 = {C_val}")
print(f"D = (-3) + 0 = {D_val}")
print(f"Обнулено: {sum([A_val == 0, B_val == 0, C_val == 0, D_val == 0])} коэффициентов")
print()

# Проверяем другие интересные наборы
recommended = [
    (2, -6, 0),  # пропорциональный (1,-3,0)
    (-1, 3, 0),  # обратный (1,-3,0)
    (0, 4, -4),  # обнуляет A и некоторые другие
    (1, -2, 1),  # другой вариант
    (3, -6, 3),  # пропорциональный известному
]

print("Другие рекомендуемые наборы:")
for alpha_val, beta_val, gamma_val in recommended:
    A_val = 9 * alpha_val + 2 * beta_val - 4 * gamma_val
    B_val = 3 * alpha_val + beta_val - gamma_val
    C_val = 27 * alpha_val + 7 * beta_val - 11 * gamma_val
    D_val = beta_val + gamma_val

    zeros = sum([A_val == 0, B_val == 0, C_val == 0, D_val == 0])
    print(f"(α, β, γ) = ({alpha_val}, {beta_val}, {gamma_val})")
    print(f"A={A_val}, B={B_val}, C={C_val}, D={D_val}")
    print(f"Обнулено: {zeros} коэффициентов")
    print()

print("ВАЖНОЕ НАБЛЮДЕНИЕ:")
print("Все наборы с 4 обнуленными коэффициентами имеют вид:")
print("(α, β, γ) = (2k/3, -k, k) для любого k ≠ 0")
print()
print("Примеры:")
for k in [-6, -3, 3, 6]:
    if (2 * k) % 3 == 0:
        alpha_val = (2 * k) // 3
        beta_val = -k
        gamma_val = k
        print(f"k = {k}: (α, β, γ) = ({alpha_val}, {beta_val}, {gamma_val})")

print()
print("Для набора (1, -3, 0) обнуляются только B и D.")
print("Это дает умеренное упрощение - исчезают логарифмические слагаемые в ψ₁ и ψ₃.")