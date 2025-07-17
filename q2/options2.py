import sympy

# Объявляем x как символьную переменную
x = sympy.symbols('x')

# Заданные значения параметров
alpha4 = -4
beta4 = 20
gamma4 = 1
alpha5 = 1
beta5 = 1
gamma5 = 1

# Вычисление psi1^(2)
# Числитель
numerator_psi1 = (
    6 * (3 * alpha4 + beta4 - gamma4 - 9 * alpha5 - 2 * beta5 + 4 * gamma5) * x**3 +
    3 * (3 * alpha4 - 2 * gamma4 + 18 * alpha5 + 4 * beta5 - 8 * gamma5) * x**2 +
    2 * (2 * gamma4 + 9 * alpha5 + 2 * beta5 - 4 * gamma5) * x +
    2 * (gamma4 + beta5 + 7 * gamma5)
)
psi1_2 = sympy.exp(-2 * x) * numerator_psi1 / 18

# Вычисление psi2^(2)
# Числитель
numerator_psi2 = (
    (-27 * alpha4 - 7 * beta4 + 11 * gamma4 + 27 * alpha5 + 6 * beta5 - 12 * gamma5) * x**3 +
    3 * (beta4 + gamma4 - 27 * alpha5 - 6 * beta5 + 12 * gamma5) * x**2 -
    6 * gamma4 * x -
    6 * gamma5
)
psi2_2 = sympy.exp(-2 * x) * numerator_psi2 / 6

# Вычисление psi3^(2)
# Числитель
numerator_psi3 = (
    beta4 + gamma4 - 27 * alpha5 - 6 * beta5 + 12 * gamma5
)
psi3_2 = sympy.exp(-2 * x) * numerator_psi3 * x**3 / 6

print("Выражение для psi1^(2) после подстановки параметров:")
print(psi1_2)
print("\nВыражение для psi2^(2) после подстановки параметров:")
print(psi2_2)
print("\nВыражение для psi3^(2) после подстановки параметров:")
print(psi3_2)