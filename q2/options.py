import sympy as sp

# Определяем символы
x, alpha4, beta4, gamma4, alpha5, beta5, gamma5 = sp.symbols('x alpha4 beta4 gamma4 alpha5 beta5 gamma5')

# Определяем уравнения системы
psi1 = (
    sp.exp(-2 * x)
    * (
        6 * (3 * alpha4 + beta4 - gamma4 - 9 * alpha5 - 2 * beta5 + 4 * gamma5) * x**3
        + 3 * (3 * alpha4 - 2 * gamma4 + 18 * alpha5 + 4 * beta5 - 8 * gamma5) * x**2
        + 2 * (2 * gamma4 + 9 * alpha5 + 2 * beta5 - 4 * gamma5) * x
        + 2 * (gamma4 + beta5 + 7 * gamma5)
    )
    / 18
)

psi2 = (
    sp.exp(-2 * x)
    * (
        (-27 * alpha4 - 7 * beta4 + 11 * gamma4 + 27 * alpha5 + 6 * beta5 - 12 * gamma5) * x**3
        + 3 * (beta4 + gamma4 - 27 * alpha5 - 6 * beta5 + 12 * gamma5) * x**2
        - 6 * gamma4 * x
        - 6 * gamma5
    )
    / 6
)

psi3 = (
    sp.exp(-2 * x)
    * x**3
    * (beta4 + gamma4 - 27 * alpha5 - 6 * beta5 + 12 * gamma5)
    / 6
)

# Задаем значения параметров
params = {
    alpha4: -6,
    beta4: 27,
    gamma4: 0,
    alpha5: -1,
    beta5: 9,
    gamma5: 0,
}

# Подставляем параметры в уравнения
psi1_substituted = psi1.subs(params)
psi2_substituted = psi2.subs(params)
psi3_substituted = psi3.subs(params)

# Упрощаем выражения
psi1_simplified = sp.simplify(psi1_substituted)
psi2_simplified = sp.simplify(psi2_substituted)
psi3_simplified = sp.simplify(psi3_substituted)

# Выводим результат
print("Уравнение psi1 после подстановки:")
sp.pprint(psi1_simplified)
print("\nУравнение psi2 после подстановки:")
sp.pprint(psi2_simplified)
print("\nУравнение psi3 после подстановки:")
sp.pprint(psi3_simplified)