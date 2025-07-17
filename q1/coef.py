from sympy import symbols, solve, Eq

# Определяем символы
alpha1, beta1, gamma1, alpha2, beta2, gamma2, alpha3, beta3, gamma3 = symbols('alpha1 beta1 gamma1 alpha2 beta2 gamma2 alpha3 beta3 gamma3')
a2, b2, c2, a1, b1, c1, a0, b0, c0 = symbols('a2 b2 c2 a1 b1 c1 a0 b0 c0')

# Система для x^2
eq1 = Eq(4*a2 + 2*b2 - 2*c2, -alpha1)
eq2 = Eq(-27*a2 - 9*b2 + 11*c2, -beta1)
eq3 = Eq(b2 - c2, -gamma1)

# Решаем для a2, b2, c2
sol_x2 = solve((eq1, eq2, eq3), (a2, b2, c2))

# Система для x
eq4 = Eq(2*sol_x2[a2], 4*a1 + 2*b1 - 2*c1 + alpha2)
eq5 = Eq(2*sol_x2[b2], -27*a1 - 9*b1 + 11*c1 + beta2)
eq6 = Eq(2*sol_x2[c2], b1 - c1 + gamma2)

# Решаем для a1, b1, c1
sol_x1 = solve((eq4, eq5, eq6), (a1, b1, c1))

# Система для x^0
eq7 = Eq(sol_x1[a1], 4*a0 + 2*b0 - 2*c0 + alpha3)
eq8 = Eq(sol_x1[b1], -27*a0 - 9*b0 + 11*c0 + beta3)
eq9 = Eq(sol_x1[c1], b0 - c0 + gamma3)

# Решаем для a0, b0, c0
sol_x0 = solve((eq7, eq8, eq9), (a0, b0, c0))

# Выводим результаты
print("Коэффициенты при x^2:")
print(f"a2 = {sol_x2[a2]}")
print(f"b2 = {sol_x2[b2]}")
print(f"c2 = {sol_x2[c2]}")
print("\nКоэффициенты при x:")
print(f"a1 = {sol_x1[a1]}")
print(f"b1 = {sol_x1[b1]}")
print(f"c1 = {sol_x1[c1]}")
print("\nКоэффициенты при x^0:")
print(f"a0 = {sol_x0[a0]}")
print(f"b0 = {sol_x0[b0]}")
print(f"c0 = {sol_x0[c0]}")