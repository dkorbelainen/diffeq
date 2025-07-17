import sympy as sp
from sympy import symbols, Matrix, exp, log, integrate, simplify, Abs, diff

# Define symbolic variables
x = symbols('x', real=True, positive=True) # x > 0 for log(x) to be real
alpha0, beta0, gamma0 = symbols('alpha0 beta0 gamma0', real=True)

print("--- Проверка и вычисление для неоднородности q^(0) ---")
print("=" * 60)

# 1. Определение матрицы A и фундаментальной матрицы Phi(x)
# Матрица A (из условия)
A = Matrix([
    [4, 2, -2],
    [-27, -9, 11],
    [0, 1, -1]
])
print("Матрица A:")
print(A)
print("-" * 20)

# Фундаментальная матрица Φ(x) (БЕЗ множителя exp(-2*x), т.е. P(x))
# Это та исправленная матрица, которую мы успешно проверили
Phi_reduced = Matrix([
    [2, 6 * x - 2, 18 * x ** 2 - 12 * x - 2],
    [-3, 9 - 9 * x, 54 * x - 27 * x ** 2],
    [3, 9 * x, 27 * x ** 2]
])
print("Фундаментальная матрица Φ(x) (без exp(-2x)):")
print(Phi_reduced)
print("-" * 20)

# Полная фундаментальная матрица (для финальной проверки)
Phi_full = exp(-2 * x) * Phi_reduced

# 2. Определение неоднородности q^(0)(x)
# q^(0) = x^(-2) * exp(-2x) * [α₀, β₀, γ₀]^T
# Для системы Φ(x)c'(x) = q^(0)_reduced(x), нам нужна q^(0)_reduced без exp(-2x)
q0_reduced = Matrix([alpha0 / x**2, beta0 / x**2, gamma0 / x**2])
print("Вектор неоднородности q^(0)_reduced(x) (без exp(-2x)):")
print(q0_reduced)
print("-" * 20)

# Полный вектор неоднородности (для финальной проверки)
q0_full = exp(-2 * x) * q0_reduced


# --- ШАГ 1: Вычисление c'(x) ---
print("\n--- ШАГ 1: Вычисление c'(x) из Φ(x)c'(x) = q^(0)_reduced(x) ---")
print("=" * 60)

# Вычисляем обратную матрицу Phi_reduced
Phi_reduced_inv = Phi_reduced.inv()

# Вычисляем c'(x)
c_prime_calculated = Matrix([
     -sp.Rational(1,6) * (3*(9*alpha0 + 2*beta0 -4*gamma0) + 2*(beta0 + gamma0) * x**(-1) - 2*gamma0 * x**(-2)),
     sp.Rational(1,9) * (3*(9*alpha0 + 2*beta0 - 4*gamma0) * x**(-1) + (beta0 + gamma0) * x**(-2)),
     -sp.Rational(1,18) * (9*alpha0 + 2*beta0 - 4*gamma0) * x**(-2)
 ])

print("Вектор c'(x):")
for i in range(3):
    print(f"c_{i+1}'(x) = {c_prime_calculated[i]}")
print("-" * 20)

# --- ШАГ 2: Интегрирование c'(x) для получения c(x) ---
print("\n--- ШАГ 2: Интегрирование c'(x) для получения c(x) ---")
print("=" * 60)

c_solutions = []
for i in range(3):
    c_integrated = integrate(c_prime_calculated[i], x)
    c_simplified = simplify(c_integrated) # Упрощаем после интегрирования
    c_solutions.append(c_simplified)
    print(f"c_{i+1}(x) = {c_simplified}")
print("-" * 20)

# Вектор c(x)
c_vector = Matrix(c_solutions)

# --- ШАГ 3: Проверка интегрирования (дифференцируем c(x) и сравниваем с c'(x)) ---
print("\n--- ШАГ 3: Проверка правильности интегрирования ---")
print("=" * 60)

integration_check_passed = True
for i in range(3):
    c_from_integral_derivative = simplify(diff(c_solutions[i], x))
    difference = simplify(c_from_integral_derivative - c_prime_calculated[i])
    print(f"d/dx[c_{i+1}(x)] - c_{i+1}'(x) = {difference}")
    if difference != 0:
        integration_check_passed = False

if integration_check_passed:
    print("\n✓ Проверка интегрирования пройдена: производные c(x) точно соответствуют c'(x).")
else:
    print("\n✗ Ошибка в проверке интегрирования!")
print("-" * 20)

# --- ШАГ 4: Вычисление полного частного решения ψ^(0)(x) ---
print("\n--- ШАГ 4: Вычисление полного частного решения ψ^(0)(x) ---")
print("=" * 60)

psi0_particular = Phi_full * c_vector

print("Полное частное решение ψ^(0)(x):")
for i in range(3):
    # Разворачиваем для более читаемого вида, если возможно
    expanded_psi = sp.expand(psi0_particular[i])
    print(f"ψ_{i+1}^(0)(x) = {expanded_psi}")
print("-" * 20)

# --- ШАГ 5: Финальная проверка: ψ'^(0) = Aψ^(0) + q^(0) ---
print("\n--- ШАГ 5: Финальная проверка частного решения ---")
print("=" * 60)

psi0_prime = diff(psi0_particular, x)
check_equation = simplify(psi0_prime - A * psi0_particular - q0_full)

print("Проверяем: ψ'^(0) - Aψ^(0) - q^(0) = ")
all_zeros_final_check = True
for i in range(3):
    component_check = simplify(check_equation[i])
    print(f"Компонента {i+1}: {component_check}")
    if component_check != 0:
        all_zeros_final_check = False

if all_zeros_final_check:
    print("\n✓ ФИНАЛЬНАЯ ПРОВЕРКА УСПЕШНА: Найденное ψ^(0)(x) является правильным частным решением!")
else:
    print("\n✗ ФИНАЛЬНАЯ ПРОВЕРКА НЕУДАЧНА: Найденное ψ^(0)(x) не удовлетворяет уравнению.")
print("=" * 60)