import sympy as sp
from sympy import symbols, Eq, solve, simplify

# Определяем символы
a0, a1, a2, a3, a4 = symbols('a0 a1 a2 a3 a4')
b0, b1, b2, b3, b4 = symbols('b0 b1 b2 b3 b4')
c0, c1, c2, c3, c4 = symbols('c0 c1 c2 c3 c4')
alpha4, alpha5 = symbols('alpha4 alpha5')
beta4, beta5 = symbols('beta4 beta5')
gamma4, gamma5 = symbols('gamma4 gamma5')

print("РЕШЕНИЕ СИСТЕМ УРАВНЕНИЙ С ВЫРАЖЕНИЕМ ВСЕХ КОЭФФИЦИЕНТОВ ЧЕРЕЗ α, β, γ")
print("="*80)

# Система для x^4 - все коэффициенты равны 0
print("\n1. Система для x^4:")
print("Из системы получаем: a4 = b4 = c4 = 0")
a4_val = 0
b4_val = 0
c4_val = 0

# Система для x^0
print("\n2. Система для x^0:")
print("Из b0 = -γ5 получаем:")
b0_val = -gamma5
print(f"b0 = {b0_val}")

# Решаем для a0 и a1
eq1 = Eq(-6*a0 + a1 - 2*b0_val, alpha5)
eq2 = Eq(27*a0 + 7*b0_val + b1, beta5)

# Выражаем a0 и a1 через b1
sol_a0_a1 = solve([eq1, eq2], [a0, a1])
print(f"a0 = {simplify(sol_a0_a1[a0])}")
print(f"a1 = {simplify(sol_a0_a1[a1])}")

# Система для x^1
print("\n3. Система для x^1:")
print("Из b1 = -γ4 получаем:")
b1_val = -gamma4
print(f"b1 = {b1_val}")

# Теперь подставляем b1 = -γ4 в найденные выражения для a0 и a1
a0_val = sol_a0_a1[a0].subs(b1, b1_val)
a1_val = sol_a0_a1[a1].subs(b1, b1_val)

print(f"a0 = {simplify(a0_val)}")
print(f"a1 = {simplify(a1_val)}")

# Решаем для a2 и b2
eq1 = Eq(-6*a1_val + 2*a2 - 2*b1_val, alpha4)
eq2 = Eq(27*a1_val + 7*b1_val + 2*b2, beta4)

sol_a2_b2 = solve([eq1, eq2], [a2, b2])
a2_val = simplify(sol_a2_b2[a2])
b2_val = simplify(sol_a2_b2[b2])

print(f"a2 = {a2_val}")
print(f"b2 = {b2_val}")

# Система для x^2
print("\n4. Система для x^2:")
# Из c3 = b2/3
c3_val = simplify(b2_val/3)
print(f"c3 = {c3_val}")

# Решаем для a3 и b3
eq1 = Eq(-6*a2_val + 3*a3 - 2*b2_val, 0)
eq2 = Eq(27*a2_val + 7*b2_val + 3*b3, 0)

sol_a3_b3 = solve([eq1, eq2], [a3, b3])
a3_val = simplify(sol_a3_b3[a3])
b3_val = simplify(sol_a3_b3[b3])

print(f"a3 = {a3_val}")
print(f"b3 = {b3_val}")

# Система для x^3
print("\n5. Система для x^3:")
print("Проверяем совместность системы x^3:")

# Подставляем все найденные значения
eq1_check = -6*a3_val + 4*a4_val - 2*b3_val + 2*c3_val
eq2_check = 27*a3_val + 7*b3_val + 4*b4_val - 11*c3_val
eq3_check = -b3_val - c3_val + 4*c4_val

print(f"Уравнение 1: {simplify(eq1_check)} = 0")
print(f"Уравнение 2: {simplify(eq2_check)} = 0")
print(f"Уравнение 3: {simplify(eq3_check)} = 0")

# Проверяем, что все три уравнения дают одинаковое условие
condition = simplify(eq1_check)
print(f"\nУсловие совместности: {condition} = 0")
print(f"Это эквивалентно: {simplify(-condition/3)} = 0")

print("\n" + "="*80)
print("ОКОНЧАТЕЛЬНЫЕ ИСПРАВЛЕННЫЕ ФОРМУЛЫ:")
print("="*80)

print("\nКоэффициенты a:")
print(f"a0 = {a0_val}")
print(f"a1 = {a1_val}")
print(f"a2 = {a2_val}")
print(f"a3 = {a3_val}")
print(f"a4 = {a4_val}")

print("\nКоэффициенты b:")
print(f"b0 = {b0_val}")
print(f"b1 = {b1_val}")
print(f"b2 = {b2_val}")
print(f"b3 = {b3_val}")
print(f"b4 = {b4_val}")

print("\nКоэффициенты c:")
print(f"c3 = {c3_val}")
print(f"c4 = {c4_val}")

##############################################################3

import sympy

# Define the symbols
a4, a3, a2, a1, a0 = sympy.symbols('a_4 a_3 a_2 a_1 a_0')
b4, b3, b2, b1, b0 = sympy.symbols('b_4 b_3 b_2 b_1 b_0')
c4, c3, c2, c1, c0 = sympy.symbols('c_4 c_3 c_2 c_1 c_0')
alpha4, alpha5, beta4, beta5, gamma4, gamma5 = sympy.symbols('alpha_4 alpha_5 beta_4 beta_5 gamma_4 gamma_5')
x = sympy.symbols('x')

# Define the given values
a_0 = (beta5 + 7*gamma5 + gamma4) / 27
a_1 = alpha5 + 2 * (beta5 - 2*gamma5 + gamma4) / 9
a_2 = (3*alpha4 + 18*alpha5 + 4*beta5 - 8*gamma5 - 2*gamma4) / 6
a_3 = alpha4 - 3*alpha5 + (beta4 - 2*beta5 + 4*gamma5 - gamma4) / 3
a_4 = 0

b_0 = -gamma5
b_1 = -gamma4
b_2 = -3*beta5 + 6*gamma5 + (gamma4 + beta4 - 27*alpha5) / 2
b_3 = beta5 - 2*gamma5 - (27*alpha4 - 27*alpha5 + 7*beta4 - 11*gamma4) / 6
b_4 = 0

c_0 = 0
c_1 = 0
c_2 = 0
c_3 = 2*gamma5 - beta5 - (27*alpha5 - beta4 - gamma4) / 6
c_4 = 0

# Define the equations for psi_1^(2)
psi_1_eq1 = sympy.exp(-2*x) * (a_4 * x**4 + a_3 * x**3 + a_2 * x**2 + a_1 * x + a_0)
psi_1_eq2 = sympy.exp(-2*x) * (b_4 * x**4 + b_3 * x**3 + b_2 * x**2 + b_1 * x + b_0)
psi_1_eq3 = sympy.exp(-2*x) * (c_4 * x**4 + c_3 * x**3 + c_2 * x**2 + c_1 * x + c_0)

# Substitute the given values into the equations
psi_1_eq1_sub = psi_1_eq1.subs({a_0: a_0, a_1: a_1, a_2: a_2, a_3: a_3, a_4: a_4})
psi_1_eq2_sub = psi_1_eq2.subs({b_0: b_0, b_1: b_1, b_2: b_2, b_3: b_3, b_4: b_4})
psi_1_eq3_sub = psi_1_eq3.subs({c_0: c_0, c_1: c_1, c_2: c_2, c_3: c_3, c_4: c_4})

# Print the results
print("Equation 1:")
print(psi_1_eq1_sub)
print("\nEquation 2:")
print(psi_1_eq2_sub)
print("\nEquation 3:")
print(psi_1_eq3_sub)