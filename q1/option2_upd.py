import sympy

# Объявляем x как символьную переменную
x = sympy.symbols('x')

# Заданные значения параметров
alpha1 = 4
beta1 = -27
gamma1 = 0

alpha2 = -2
beta2 = 0
gamma2 = 0

alpha3 = 4
beta3 = -27
gamma3 = 0

# --- Вычисление psi1^(1) ---
psi1_1_numerator = (
    2 * (-alpha1 + 2 * gamma1) * x**2 +
    2 * (13 * alpha1 + 2 * beta1 - 8 * gamma1 - alpha2 + 2 * gamma2) * x -
    34 * alpha1 - 6 * beta1 + 18 * gamma1 + 13 * alpha2 + 2 * beta2 - 8 * gamma2 - 2 * alpha3 + 4 * gamma3
)
psi1_1 = psi1_1_numerator / 8

# --- Вычисление psi2^(1) ---
psi2_1_numerator = (
    2 * (-27 * alpha1 - 4 * beta1 + 10 * gamma1) * x**2 +
    2 * (27 * alpha1 + 6 * beta1 - 8 * gamma1 - 27 * alpha2 - 4 * beta2 + 10 * gamma2) * x -
    2 * beta1 - 6 * gamma1 + 27 * alpha2 + 6 * beta2 - 8 * gamma2 - 54 * alpha3 - 8 * beta3 + 20 * gamma3
)
psi2_1 = psi2_1_numerator / 16

# --- Вычисление psi3^(1) ---
psi3_1_numerator = (
    2 * (-27 * alpha1 - 4 * beta1 + 18 * gamma1) * x**2 +
    2 * (81 * alpha1 + 14 * beta1 - 44 * gamma1 - 27 * alpha2 - 4 * beta2 + 18 * gamma2) * x -
    162 * alpha1 - 30 * beta1 + 82 * gamma1 + 81 * alpha2 + 14 * beta2 - 44 * gamma2 - 54 * alpha3 - 8 * beta3 + 36 * gamma3
)
psi3_1 = psi3_1_numerator / 16

# Вывод результатов
print("Выражение для psi1^(1) после подстановки параметров:")
print(psi1_1)
print("\nВыражение для psi2^(1) после подстановки параметров:")
print(psi2_1)
print("\nВыражение для psi3^(1) после подстановки параметров:")
print(psi3_1)