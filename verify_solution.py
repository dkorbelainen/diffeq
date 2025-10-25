"""
Проверка корректности решений: подставляем y(x) в y' = Ay + q(x)
"""
from sympy import symbols, Matrix, exp, sin, cos, diff, simplify, log, Abs
from polynomial.solve import solve_polynomial_case
from resonance.solve import solve_resonance_case
from trigonometric.solve import solve_trigonometric_case
from special_point.solve import solve_special_case


def verify_solution(A, y, q, x):
    """Проверяет, что y' = Ay + q"""
    y_prime = Matrix([diff(yi, x) for yi in y])
    rhs = A * y + q
    diff_vec = simplify(y_prime - rhs)

    is_correct = all(simplify(d) == 0 for d in diff_vec)
    return is_correct, y_prime, rhs, diff_vec


def test_polynomial():
    """Тест полиномиального решения"""
    print("\n" + "="*60)
    print("ТЕСТ 1: ПОЛИНОМ")
    print("="*60)

    x = symbols('x')
    A = Matrix([[4, 2, -2], [-27, -9, 11], [0, 1, -1]])

    # q(x) = [-2x² + 2x - 4, 9x² - 9x + 27, -x² + x + 1]ᵀ
    q = Matrix([
        -2*x**2 + 2*x - 4,
        9*x**2 - 9*x + 27,
        -x**2 + x + 1
    ])

    result = solve_polynomial_case((-2, 2, -4), (9, -9, 27), (-1, 1, 1))

    # Восстанавливаем y из коэффициентов
    y = Matrix([
        result['psi1'][0]*x**2 + result['psi1'][1]*x + result['psi1'][2],
        result['psi2'][0]*x**2 + result['psi2'][1]*x + result['psi2'][2],
        result['psi3'][0]*x**2 + result['psi3'][1]*x + result['psi3'][2]
    ])

    is_correct, y_prime, rhs, diff_vec = verify_solution(A, y, q, x)

    print(f"y = {y}")
    print(f"\ny' = {y_prime}")
    print(f"\nAy + q = {simplify(rhs)}")
    print(f"\nРазность y' - (Ay + q) = {diff_vec}")
    print(f"\n✓ Решение {'КОРРЕКТНО' if is_correct else 'НЕКОРРЕКТНО'}")

    return is_correct


def test_resonance():
    """Тест резонансного решения"""
    print("\n" + "="*60)
    print("ТЕСТ 2: РЕЗОНАНС")
    print("="*60)

    x = symbols('x')
    A = Matrix([[4, 2, -2], [-27, -9, 11], [0, 1, -1]])

    # q(x) = e^(-2x) * [(-16x + 2), (56x + 1), (-8x + 1)]ᵀ
    q = exp(-2*x) * Matrix([
        -16*x + 2,
        56*x + 1,
        -8*x + 1
    ])

    result = solve_resonance_case((-16, 2), (56, 1), (-8, 1))
    y = Matrix([result['psi1'], result['psi2'], result['psi3']])

    is_correct, y_prime, rhs, diff_vec = verify_solution(A, y, q, x)

    print(f"y = {y}")
    print(f"\ny' = {simplify(y_prime)}")
    print(f"\nAy + q = {simplify(rhs)}")
    print(f"\nРазность y' - (Ay + q) = {simplify(diff_vec)}")
    print(f"\n✓ Решение {'КОРРЕКТНО' if is_correct else 'НЕКОРРЕКТНО'}")

    return is_correct


def test_trigonometric():
    """Тест тригонометрического решения"""
    print("\n" + "="*60)
    print("ТЕСТ 3: ТРИГОНОМЕТРИЯ")
    print("="*60)

    x = symbols('x')
    A = Matrix([[4, 2, -2], [-27, -9, 11], [0, 1, -1]])

    # q(x) = [(48x + 4)sin(x), (-168x - 32)sin(x), (39x + 0)sin(x)]ᵀ
    q = Matrix([
        (48*x + 4)*sin(x),
        (-168*x - 32)*sin(x),
        39*x*sin(x)
    ])

    result = solve_trigonometric_case((48, 4), (-168, -32), (39, 0))
    y = Matrix([result['psi1'], result['psi2'], result['psi3']])

    is_correct, y_prime, rhs, diff_vec = verify_solution(A, y, q, x)

    print(f"y = {y}")
    print(f"\ny' = {simplify(y_prime)}")
    print(f"\nAy + q = {simplify(rhs)}")
    print(f"\nРазность y' - (Ay + q) = {simplify(diff_vec)}")
    print(f"\n✓ Решение {'КОРРЕКТНО' if is_correct else 'НЕКОРРЕКТНО'}")

    return is_correct


def test_special():
    """Тест особой точки"""
    print("\n" + "="*60)
    print("ТЕСТ 4: ОСОБАЯ ТОЧКА")
    print("="*60)

    x = symbols('x', real=True, positive=True)
    A = Matrix([[4, 2, -2], [-27, -9, 11], [0, 1, -1]])

    # q(x) = e^(-2x) * x^(-2) * [2, -3, 3]ᵀ
    q = exp(-2*x) / x**2 * Matrix([2, -3, 3])

    result = solve_special_case(2, -3, 3, verbose=False)
    y = Matrix([result['psi1'], result['psi2'], result['psi3']])

    is_correct, y_prime, rhs, diff_vec = verify_solution(A, y, q, x)

    print(f"y = {y}")
    print(f"\ny' = {simplify(y_prime)}")
    print(f"\nAy + q = {simplify(rhs)}")
    print(f"\nРазность y' - (Ay + q) = {simplify(diff_vec)}")
    print(f"\n✓ Решение {'КОРРЕКТНО' if is_correct else 'НЕКОРРЕКТНО'}")

    return is_correct


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ВЕРИФИКАЦИЯ ВСЕХ РЕШЕНИЙ")
    print("="*60)

    results = []
    results.append(("Полином", test_polynomial()))
    results.append(("Резонанс", test_resonance()))
    results.append(("Тригонометрия", test_trigonometric()))
    results.append(("Особая точка", test_special()))

    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    for name, is_correct in results:
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"{status:10} {name}")

    all_correct = all(r[1] for r in results)
    print("\n" + "="*60)
    if all_correct:
        print("ВСЕ РЕШЕНИЯ КОРРЕКТНЫ! ✓")
    else:
        print("НАЙДЕНЫ ОШИБКИ! ✗")
    print("="*60 + "\n")

