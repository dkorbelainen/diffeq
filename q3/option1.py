import sympy

# Определяем символьные переменные
x = sympy.Symbol('x')
alpha_6, beta_6, gamma_6 = sympy.symbols('alpha_6 beta_6 gamma_6')
alpha_7, beta_7, gamma_7 = sympy.symbols('alpha_7 beta_7 gamma_7')

# Задаём параметры
params = {
    alpha_6: 48,
    beta_6: -168,
    gamma_6: 39,
    alpha_7: 4,
    beta_7: -32,
    gamma_7: 0
}

# Определяем уравнения
y1_expr = (sympy.sin(x) * (-573*alpha_6 - 124*beta_6 + 208*gamma_6 + 520*alpha_7 + 110*beta_7 - 70*gamma_7 + 10*x*(52*alpha_6 + 11*beta_6 - 7*gamma_6)) -
           sympy.cos(x) * (536*alpha_6 + 68*beta_6 - 356*gamma_6 - 265*alpha_7 - 20*beta_7 + 240*gamma_7 - 5*x*(53*alpha_6 + 4*beta_6 - 48*gamma_6))) / 625

y2_expr = (sympy.sin(x) * (1107*alpha_6 + 191*beta_6 - 472*gamma_6 - 1755*alpha_7 - 215*beta_7 + 705*gamma_7 - 5*x*(351*alpha_6 + 43*beta_6 - 141*gamma_6)) -
           sympy.cos(x) * (1026*alpha_6 + 238*beta_6 - 346*gamma_6 - 1215*alpha_7 - 245*beta_7 + 440*gamma_7 - 5*x*(243*alpha_6 + 49*beta_6 - 88*gamma_6))) / 625

y3_expr = (sympy.sin(x) * (-567*alpha_6 - 146*beta_6 + 157*gamma_6 - 270*alpha_7 + 15*beta_7 + 445*gamma_7 - 5*x*(54*alpha_6 - 3*beta_6 - 89*gamma_6)) -
           sympy.cos(x) * (1944*alpha_6 + 322*beta_6 - 1074*gamma_6 - 1485*alpha_7 - 230*beta_7 + 885*gamma_7 - 5*x*(297*alpha_6 + 46*beta_6 - 177*gamma_6))) / 625

# Заменяем параметры на их значения в каждом уравнении
y1_substituted = y1_expr.subs(params)
y2_substituted = y2_expr.subs(params)
y3_substituted = y3_expr.subs(params)

# Упрощаем выражения
y1_simplified = sympy.simplify(y1_substituted)
y2_simplified = sympy.simplify(y2_substituted)
y3_simplified = sympy.simplify(y3_substituted)

# Выводим результат
print("Уравнение y_1 с подставленными значениями:")
print(f"y_1 = {y1_simplified}")
print("-" * 50)
print("Уравнение y_2 с подставленными значениями:")
print(f"y_2 = {y2_simplified}")
print("-" * 50)
print("Уравнение y_3 с подставленными значениями:")
print(f"y_3 = {y3_simplified}")