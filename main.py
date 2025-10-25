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
        print("="*70)

        # Визуализация и анализ
        print("\n4. Визуализация и анализ (опционально):")
        print("   [1] Построить графики")
        print("   [2] Численный анализ (сравнение с scipy)")
        print("   [3] Пропустить")
        choice = input("   Выбор: ").strip()

        if choice in ["1", "2"]:
            import matplotlib
            matplotlib.use('Agg')
            from pathlib import Path
            from visualization import plot_solution

            Path("results").mkdir(exist_ok=True)

            if choice == "1":
                # Просто визуализация
                x_min = float(input("\n   Диапазон x от: ").strip() or "0")
                x_max = float(input("   Диапазон x до: ").strip() or "3")

                plot_solution(psi, x_range=(x_min, x_max),
                            title="Ваше решение ЛНС",
                            save_path="results/custom_solution.png")
                print(f"\n   ✓ График сохранен: results/custom_solution.png")

            elif choice == "2":
                # Полный анализ
                from numerical_analysis import analyze_solution
                import numpy as np
                from sympy import lambdify

                x_min = float(input("\n   Диапазон x от: ").strip() or "0")
                x_max = float(input("   Диапазон x до: ").strip() or "3")

                # Начальные условия
                print("\n   Начальные условия y(x_min):")
                y0 = []
                for i in range(n):
                    val = float(input(f"     y[{i+1}]({x_min}) = ").strip() or "0")
                    y0.append(val)
                y0 = np.array(y0)

                # Конвертируем A и q в numpy
                A_numpy = np.array(A.tolist(), dtype=float)

                # Создаем функцию q(x)
                q_lambdas = [lambdify(x, q_i, modules=['numpy']) for q_i in q_components]
                def q_func(x_val):
                    return np.array([f(x_val) for f in q_lambdas])

                # Анализ
                analyze_solution(psi, A_numpy, q_func, y0,
                               x_range=(x_min, x_max),
                               solution_name="Ваше решение",
                               save_prefix="results/custom")

                print(f"\n   ✓ Результаты сохранены в results/")

        print("\n")

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

    # Визуализация (опционально)
    print("\n" + "="*70)
    print("Построить графики решения? [y/N]: ", end='')
    choice = input().strip().lower()

    if choice == 'y':
        import matplotlib
        matplotlib.use('Agg')
        from pathlib import Path
        from visualization import plot_solution

        Path("results").mkdir(exist_ok=True)

        # Объединяем все частные решения
        combined_solution = {
            'psi1': psi0['psi1'] + psi1['psi1'][0] + psi2['psi1'] + psi3['psi1'],
            'psi2': psi0['psi2'] + psi1['psi2'][0] + psi2['psi2'] + psi3['psi2'],
            'psi3': psi0['psi3'] + psi1['psi3'][0] + psi2['psi3'] + psi3['psi3']
        }

        # Исправляем полиномы (конвертируем из tuple)
        from sympy import symbols as sym
        x_sym = sym('x')
        combined_solution['psi1'] = psi0['psi1'] + psi1['psi1'][0]*x_sym**2 + psi1['psi1'][1]*x_sym + psi1['psi1'][2] + psi2['psi1'] + psi3['psi1']
        combined_solution['psi2'] = psi0['psi2'] + psi1['psi2'][0]*x_sym**2 + psi1['psi2'][1]*x_sym + psi1['psi2'][2] + psi2['psi2'] + psi3['psi2']
        combined_solution['psi3'] = psi0['psi3'] + psi1['psi3'][0]*x_sym**2 + psi1['psi3'][1]*x_sym + psi1['psi3'][2] + psi2['psi3'] + psi3['psi3']

        plot_solution(combined_solution, x_range=(0.1, 3),
                     title=f"Вариант {variant_num}: Полное решение ЛНС",
                     save_path=f"results/variant{variant_num}_solution.png")
        print(f"\n✓ График сохранен: results/variant{variant_num}_solution.png")
        print("="*70 + "\n")


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
