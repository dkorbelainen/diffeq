import sympy as sp
from sympy import symbols, Matrix, exp, log, simplify, expand, collect, factor

# Определяем символы
x, alpha0, beta0, gamma0 = symbols('x alpha_0 beta_0 gamma_0')

print("Подстановка c(x) в формулы для ψ^(0)(x)")
print("="*60)

# Найденные выражения для c(x)
c1 = -(3*x*(9*alpha0 + 2*beta0 - 4*gamma0) + 2*(beta0 + gamma0)*log(abs(x)) + 2*gamma0*x**(-1))/6
c2 = (3*(9*alpha0 + 2*beta0 - 4*gamma0)*log(abs(x)) - (beta0 + gamma0)*x**(-1))/9
c3 = (9*alpha0 + 2*beta0 - 4*gamma0)*x**(-1)/18

print("Найденные выражения для c(x):")
print(f"c_1(x) = {c1}")
print(f"c_2(x) = {c2}")
print(f"c_3(x) = {c3}")
print()

# Формулы для ψ^(0)(x)
print("Формулы для ψ^(0)(x):")
print("ψ_1^(0) = e^(-2x) * [2c_1(x) + c_2(x)(6x - 2) + c_3(x)(18x^2 - 12x - 2)]")
print("ψ_2^(0) = e^(-2x) * [-3c_1(x) + c_2(x)(-9x + 9) + c_3(x)(-27x^2 + 54x)]")
print("ψ_3^(0) = e^(-2x) * [3c_1(x) + 9c_2(x)x + 27c_3(x)x^2]")
print()

# Вычисляем каждую компоненту
print("Вычисление компонент:")
print("="*40)

# ψ_1^(0)
print("ψ_1^(0):")
print("Подставляем в: 2c_1(x) + c_2(x)(6x - 2) + c_3(x)(18x^2 - 12x - 2)")
print()

term1_1 = 2*c1
term1_2 = c2*(6*x - 2)
term1_3 = c3*(18*x**2 - 12*x - 2)

print(f"2c_1(x) = {simplify(term1_1)}")
print(f"c_2(x)(6x - 2) = {simplify(term1_2)}")
print(f"c_3(x)(18x^2 - 12x - 2) = {simplify(term1_3)}")
print()

psi1_inner = term1_1 + term1_2 + term1_3
psi1_inner_simplified = simplify(psi1_inner)

print(f"Сумма = {psi1_inner_simplified}")
print()

# Приведем к удобному виду
psi1_inner_expanded = expand(psi1_inner_simplified)
print(f"Развернутая сумма = {psi1_inner_expanded}")
print()

# Соберем по степеням x и логарифмам
psi1_collected = collect(psi1_inner_expanded, [x, log(abs(x))])
print(f"Приведенная к удобному виду:")
print(f"ψ_1^(0) внутренняя часть = {psi1_collected}")
print()

print("="*40)

# ψ_2^(0)
print("ψ_2^(0):")
print("Подставляем в: -3c_1(x) + c_2(x)(-9x + 9) + c_3(x)(-27x^2 + 54x)")
print()

term2_1 = -3*c1
term2_2 = c2*(-9*x + 9)
term2_3 = c3*(-27*x**2 + 54*x)

print(f"-3c_1(x) = {simplify(term2_1)}")
print(f"c_2(x)(-9x + 9) = {simplify(term2_2)}")
print(f"c_3(x)(-27x^2 + 54x) = {simplify(term2_3)}")
print()

psi2_inner = term2_1 + term2_2 + term2_3
psi2_inner_simplified = simplify(psi2_inner)

print(f"Сумма = {psi2_inner_simplified}")
print()

psi2_inner_expanded = expand(psi2_inner_simplified)
print(f"Развернутая сумма = {psi2_inner_expanded}")
print()

psi2_collected = collect(psi2_inner_expanded, [x, log(abs(x))])
print(f"Приведенная к удобному виду:")
print(f"ψ_2^(0) внутренняя часть = {psi2_collected}")
print()

print("="*40)

# ψ_3^(0)
print("ψ_3^(0):")
print("Подставляем в: 3c_1(x) + 9c_2(x)x + 27c_3(x)x^2")
print()

term3_1 = 3*c1
term3_2 = 9*c2*x
term3_3 = 27*c3*x**2

print(f"3c_1(x) = {simplify(term3_1)}")
print(f"9c_2(x)x = {simplify(term3_2)}")
print(f"27c_3(x)x^2 = {simplify(term3_3)}")
print()

psi3_inner = term3_1 + term3_2 + term3_3
psi3_inner_simplified = simplify(psi3_inner)

print(f"Сумма = {psi3_inner_simplified}")
print()

psi3_inner_expanded = expand(psi3_inner_simplified)
print(f"Развернутая сумма = {psi3_inner_expanded}")
print()

psi3_collected = collect(psi3_inner_expanded, [x, log(abs(x))])
print(f"Приведенная к удобному виду:")
print(f"ψ_3^(0) внутренняя часть = {psi3_collected}")
print()

print("="*60)
print("ОКОНЧАТЕЛЬНЫЙ РЕЗУЛЬТАТ:")
print("="*60)

print("ψ^(0)(x) = e^(-2x) * [внутренние части]")
print()
print(f"ψ_1^(0) = e^(-2x) * [{psi1_collected}]")
print()
print(f"ψ_2^(0) = e^(-2x) * [{psi2_collected}]")
print()
print(f"ψ_3^(0) = e^(-2x) * [{psi3_collected}]")
print()

# Дополнительно: попробуем факторизовать результаты
print("="*60)
print("ПОПЫТКА ФАКТОРИЗАЦИИ:")
print("="*60)

# Определим общий множитель
common_factor = 9*alpha0 + 2*beta0 - 4*gamma0
print(f"Общий множитель во многих слагаемых: {common_factor}")
print()

# Попробуем выделить этот множитель
psi1_factored = collect(psi1_collected, common_factor)
psi2_factored = collect(psi2_collected, common_factor)
psi3_factored = collect(psi3_collected, common_factor)

print(f"ψ_1^(0) (факторизованная) = e^(-2x) * [{psi1_factored}]")
print()
print(f"ψ_2^(0) (факторизованная) = e^(-2x) * [{psi2_factored}]")
print()
print(f"ψ_3^(0) (факторизованная) = e^(-2x) * [{psi3_factored}]")
print()

# Проверим размерности
print("="*60)
print("ПРОВЕРКА РАЗМЕРНОСТЕЙ:")
print("="*60)

print("Все слагаемые должны иметь одинаковые размерности.")
print("Проверим степени x в различных слагаемых:")

# Разложим каждое выражение и проверим степени x
from sympy import Poly

def analyze_powers(expr, var):
    """Анализирует степени переменной в выражении"""
    try:
        # Удаляем логарифмы для анализа полиномиальной части
        expr_no_log = expr.subs(log(abs(var)), 0)
        if expr_no_log.is_polynomial(var):
            poly = Poly(expr_no_log, var)
            return poly.degree()
        else:
            return "Содержит нецелые степени"
    except:
        return "Сложная структура"

print(f"ψ_1^(0): степень x = {analyze_powers(psi1_collected, x)}")
print(f"ψ_2^(0): степень x = {analyze_powers(psi2_collected, x)}")
print(f"ψ_3^(0): степень x = {analyze_powers(psi3_collected, x)}")