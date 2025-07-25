import sympy as sp

# Определение символьных переменных
x = sp.symbols('x')
alpha4, alpha5, beta4, beta5, gamma4, gamma5 = sp.symbols('alpha_4 alpha_5 beta_4 beta_5 gamma_4 gamma_5')

# Матрица A
A = sp.Matrix([[4, 2, -2], [-27, -9, 11], [0, 1, -1]])

a_2 = (3\alpha_4 + \beta_4 - 3\gamma_4) / 3
b_2 = -3\gamma_4
c_2 = -\gamma_4
a_1 = (\beta_4 + \gamma_4 - 12\alpha_5 - 4\beta_5 + 8\gamma_5) /
12 b_1 = -\gamma_4
a_0 = (\beta_4 + \gamma_4 - 42\alpha_5 - 8\beta_5 + 44\gamma_5) / 72
b_0 = -\gamma_5

# Коэффициенты вашего решения
a4 = (-9*alpha4 - 2*beta4 + 8*gamma4)/12
a3 = (3*alpha4 + beta4 - 3*gamma4 - 9*alpha5 - 2*beta5 + 4*gamma5)/3
a2 = (3alpha4 + beta4 - 3*gamma4)/3
a1 = (beta4 + gamma4 - 24*alpha5 - 8*beta5 + 16*gamma5)/12
a0 = (beta4 + gamma4 - 42*alpha5 - 8*beta5 + 44*gamma5)/72

b4 = (9*alpha4 + 2*beta4 - 8*gamma4)/8
b3 = (-27*alpha4 - 7*beta4 + 25*gamma4 + 27*alpha5 + 6*beta5 - 12*gamma5)/6
b2 = -3gamma4
b1 = -gamma4
b0 = -gamma5

c4 = (-9*alpha4 - 2*beta4 + 8*gamma4)/8
c3 = (beta4 - gamma4 - 27*alpha5 - 6*beta5 + 12*gamma5)/6

# Частное решение psi^{(2)}(x)
psi1 = sp.exp(-2*x) * (a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)
psi2 = sp.exp(-2*x) * (b4*x**4 + b3*x**3 + b2*x**2 + b1*x + b0)
psi3 = sp.exp(-2*x) * (c4*x**4 + c3*x**3)
psi = sp.Matrix([psi1, psi2, psi3])

# Вычисление производной psi'(x)
psi_diff = psi.diff(x)

# Вычисление A * psi(x)
A_psi = A * psi

# Определение q^{(2)}(x)
q2 = sp.exp(-2*x) * sp.Matrix([alpha4*x + alpha5, beta4*x + beta5, gamma4*x + gamma5])

# Вычисление правой части A * psi(x) + q^{(2)}(x)
right_side = A_psi + q2

# Проверка равенства: psi'(x) - [A * psi(x) + q^{(2)}(x)]
difference = psi_diff - right_side

# Упрощение разности
simplified_difference = sp.simplify(difference)

# Вывод результата
if simplified_difference == sp.Matrix([0, 0, 0]):
    print("Решение верно: psi'(x) = A * psi(x) + q^{(2)}(x)")
else:
    print("Решение неверно. Разность psi'(x) - [A * psi(x) + q^{(2)}(x)]:")
    print(simplified_difference)