"""
Решение ЛНС: y' = Ay + q(x)

Использование:
    python main.py           # Вариант 1 (готовый пример)
    python main.py 2         # Вариант 2 (готовый пример)
    python main.py --custom  # Решить свою систему
"""

from fundamental_matrix import verify_fundamental_matrix
from special_point.solve import solve_special_case
from polynomial.solve import solve_polynomial_case, format_polynomial
from resonance.solve import solve_resonance_case
from trigonometric.solve import solve_trigonometric_case
from sympy import symbols, Matrix, exp, simplify, integrate, sympify
import sys


def variant_1():
    """Вариант 1: максимально простые решения"""
    return {
        'special': (2, -3, 3),
        'polynomial': ((-2, 2, -4), (9, -9, 27), (-1, 1, 1)),
        'resonance': ((-16, 2), (56, 1), (-8, 1)),
        'trigonometric': ((48, 4), (-168, -32), (39, 0))
    }


def variant_2():
    """Вариант 2: альтернативные простые решения"""
    return {
        'special': (0, 2, 1),
        'polynomial': ((4, -2, 4), (-27, 0, -27), (0, 0, 0)),
        'resonance': ((-6, -1), (27, 9), (0, 0)),
        'trigonometric': ((10, 6), (-39, 0), (13, -1))
    }


def solve_custom_system(A, Phi, q, x):
    """
    Метод вариации постоянных для произвольной системы y' = Ay + q(x)
    """
    print("\n" + "="*60)
    print("  МЕТОД ВАРИАЦИИ ПОСТОЯННЫХ")
    print("="*60)

    # Проверка Φ(x)
    result = verify_fundamental_matrix(A, Phi, x, verbose=True)
    if not (result['is_valid'] and result['is_nonsingular']):
        raise ValueError("❌ Фундаментальная матрица некорректна!")

    # c'(x) = Φ⁻¹(x)·q(x)
    print("\nШаг 1: Вычисление c'(x) = Φ⁻¹(x)·q(x)")
    Phi_inv = Phi.inv()
    c_prime = Phi_inv * q
    for i, expr in enumerate(c_prime, 1):
        print(f"  c'_{i}(x) = {simplify(expr)}")

    # Интегрирование
    print("\nШаг 2: Интегрирование c(x) = ∫c'(x)dx")
    c = Matrix([integrate(expr, x) for expr in c_prime])
    for i, expr in enumerate(c, 1):
        print(f"  c_{i}(x) = {simplify(expr)}")

    # Частное решение
    print("\nШаг 3: Частное решение ψ(x) = Φ(x)·c(x)")
    psi = Phi * c
    result = {}
    n = A.shape[0]
    for i in range(n):
        result[f'psi{i+1}'] = simplify(psi[i])
        print(f"  ψ_{i+1}(x) = {result[f'psi{i+1}']}")

    return result


def custom_mode():
    """Интерактивный режим для своей системы"""
    print("\n" + "="*70)
    print("  РЕШЕНИЕ СВОЕЙ СИСТЕМЫ ЛНС")
    print("="*70)

    x = symbols('x')

    # Матрица A
    print("\n1. Матрица A:")
    print("   [1] Ввести вручную")
    print("   [2] Использовать из примера")
    choice = input("   Выбор: ").strip()

    if choice == "2":
        A = Matrix([[4, 2, -2], [-27, -9, 11], [0, 1, -1]])
        n = 3
        print(f"   ✓ A = {A}")
    else:
        n = int(input("   Размерность (2 или 3): "))
        print(f"   Введите {n} строк (элементы через запятую):")
        rows = []
        for i in range(n):
            row = [sympify(x.strip()) for x in input(f"     Строка {i+1}: ").split(',')]
            rows.append(row)
        A = Matrix(rows)

    print(f"   ✓ Собственные числа: {A.eigenvals()}")

    # Фундаментальная матрица Φ(x)
    print("\n2. Фундаментальная матрица Φ(x):")
    print("   ВАЖНО: Должна удовлетворять Φ'(x) = A·Φ(x)")
    print("   [1] Ввести вручную")
    print("   [2] Использовать из примера (только если A из примера)")
    choice = input("   Выбор: ").strip()

    if choice == "2" and n == 3:
        Phi = exp(-2*x) * Matrix([
            [2, 6*x - 2, 18*x**2 - 12*x - 2],
            [-3, -9*x + 9, -27*x**2 + 54*x],
            [3, 9*x, 27*x**2]
        ])
        print("   ✓ Используется Φ(x) из примера")
    else:
        print(f"   Введите {n} строк Φ(x) (используйте: x, exp(...), sin(...)):")
        rows = []
        for i in range(n):
            row = [sympify(x.strip()) for x in input(f"     Строка {i+1}: ").split(',')]
            rows.append(row)
        Phi = Matrix(rows)

    # Неоднородность q(x)
    print("\n3. Неоднородность q(x):")
    print("   Примеры: x, x**2, exp(-2*x), sin(x), x*exp(-2*x)")
    q_components = []
    for i in range(n):
        q_i = sympify(input(f"   q[{i+1}](x) = "))
        q_components.append(q_i)
    q = Matrix(q_components)
    print(f"   ✓ q(x) = {q}")

    # Решение
    try:
        psi = solve_custom_system(A, Phi, q, x)

        print("\n" + "="*70)
        print("  ОБЩЕЕ РЕШЕНИЕ: y(x) = Φ(x)·c + ψ(x)")
        print("="*70)
        for i in range(n):
            homo = " + ".join([f"Φ[{i},{j}]·c_{j+1}" for j in range(n)])
            print(f"\ny_{i+1}(x) = {homo}")
            print(f"         + {psi[f'psi{i+1}']}")

        print("\nРешение найдено!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}\n")


def predefined_mode(variant_num):
    """Режим готовых примеров (варианты 1 и 2)"""
    params = variant_1() if variant_num == 1 else variant_2()

    x = symbols('x')
    A = Matrix([[4, 2, -2], [-27, -9, 11], [0, 1, -1]])
    Phi = exp(-2*x) * Matrix([
        [2, 6*x - 2, 18*x**2 - 12*x - 2],
        [-3, -9*x + 9, -27*x**2 + 54*x],
        [3, 9*x, 27*x**2]
    ])

    print("\n" + "="*70)
    print(f"  РЕШЕНИЕ ЛНС: y' = Ay + q(x)  [ВАРИАНТ {variant_num}]")
    print("="*70)

    # 1. Фундаментальная матрица
    verify_fundamental_matrix(A, Phi, x, verbose=True)

    # 2. Особая точка
    print("\n" + "="*60)
    print("  ОСОБАЯ ТОЧКА: x⁻²e⁻²ˣ")
    print("="*60)
    alpha0, beta0, gamma0 = params['special']
    print(f"Параметры: α₀={alpha0}, β₀={beta0}, γ₀={gamma0}")
    psi0 = solve_special_case(alpha0, beta0, gamma0, verbose=False)
    print(f"Константы: K={psi0['K']}, L={psi0['L']}")
    for i in range(1, 4):
        print(f"ψ_{i}⁰(x) = {psi0[f'psi{i}']}")

    # 3. Полином
    print("\n" + "="*60)
    print("  ПОЛИНОМ: ax² + bx + c")
    print("="*60)
    alpha, beta, gamma = params['polynomial']
    print(f"Параметры: α={alpha}, β={beta}, γ={gamma}")
    psi1 = solve_polynomial_case(alpha, beta, gamma)
    for i in range(1, 4):
        print(f"ψ_{i}¹(x) = {format_polynomial(psi1[f'psi{i}'])}")

    # 4. Резонанс
    print("\n" + "="*60)
    print("  РЕЗОНАНС: e⁻²ˣ·(ax + b)")
    print("="*60)
    alpha, beta, gamma = params['resonance']
    print(f"Параметры: α={alpha}, β={beta}, γ={gamma}")
    print("λ = -2 совпадает с собственным числом")
    psi2 = solve_resonance_case(alpha, beta, gamma)
    for i in range(1, 4):
        print(f"ψ_{i}²(x) = {psi2[f'psi{i}']}")

    # 5. Тригонометрия
    print("\n" + "="*60)
    print("  ТРИГОНОМЕТРИЯ: (ax+b)sin(x)")
    print("="*60)
    alpha, beta, gamma = params['trigonometric']
    print(f"Параметры: α={alpha}, β={beta}, γ={gamma}")
    psi3 = solve_trigonometric_case(alpha, beta, gamma)
    for i in range(1, 4):
        print(f"ψ_{i}³(x) = {psi3[f'psi{i}']}")

    # 6. ИТОГОВЫЙ ОТВЕТ
    print("\n" + "="*70)
    print("  ИТОГОВЫЙ ОТВЕТ")
    print("="*70)
    print("\nОбщее решение ЛНС:")
    print("  y(x) = Φ(x)·c + ψ⁰(x) + ψ¹(x) + ψ²(x) + ψ³(x)")
    print("\nгде c = [c₁, c₂, c₃]ᵀ — произвольные постоянные")
    print("\nФундаментальная матрица:")
    print(f"  Φ(x) = e^(-2x) × {Phi / exp(-2*x)}")

    print("\nКомпоненты решения:")
    print("-" * 60)

    phi_components = [
        "e^(-2x) · (2c₁ + (6x-2)c₂ + (18x²-12x-2)c₃)",
        "e^(-2x) · (-3c₁ + (-9x+9)c₂ + (-27x²+54x)c₃)",
        "e^(-2x) · (3c₁ + 9xc₂ + 27x²c₃)"
    ]

    for i in range(1, 4):
        print(f"\ny_{i}(x) = {phi_components[i-1]}")
        print(f"         + {psi0[f'psi{i}']}")
        print(f"         + {format_polynomial(psi1[f'psi{i}'])}")
        print(f"         + {psi2[f'psi{i}']}")
        print(f"         + {psi3[f'psi{i}']}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--custom":
        custom_mode()
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        variant_num = int(sys.argv[1])
        if variant_num in [1, 2]:
            predefined_mode(variant_num)
        else:
            show_help()
    elif len(sys.argv) == 1:
        predefined_mode(1)
    else:
        show_help()


def show_help():
    print("\n" + "="*50)
    print("  Решение ЛНС: y' = Ay + q(x)")
    print("="*50)
    print("\nИспользование:")
    print("  python main.py          → Вариант 1 (пример)")
    print("  python main.py 2        → Вариант 2 (пример)")
    print("  python main.py --custom → Своя система")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
