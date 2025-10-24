"""Проверка фундаментальной матрицы Φ(x) для y' = Ay"""

from sympy import symbols, Matrix, exp, diff, simplify


def verify_fundamental_matrix(A, Phi, x, verbose=True):
    """Проверяет Φ'(x) = AΦ(x) и det(Φ) ≠ 0"""
    Phi_prime = diff(Phi, x)
    A_Phi = A * Phi
    difference = simplify(Phi_prime - A_Phi)
    is_valid = all(simplify(element) == 0 for element in difference)
    det_Phi = simplify(Phi.det())
    is_nonsingular = det_Phi != 0
    eigenvals = A.eigenvals()

    if verbose:
        print("\n" + "="*60)
        print("  ФУНДАМЕНТАЛЬНАЯ МАТРИЦА")
        print("="*60)
        print(f"✓ Φ'(x) = AΦ(x): {is_valid}")
        print(f"✓ det(Φ(x)) = {det_Phi}")
        print(f"✓ Собственные числа A: {eigenvals}")
        if is_valid and is_nonsingular:
            print("\nФундаментальная матрица корректна")

    return {'is_valid': is_valid, 'is_nonsingular': is_nonsingular,
            'determinant': det_Phi, 'eigenvalues': eigenvals}


if __name__ == "__main__":
    x = symbols('x')
    A = Matrix([[4, 2, -2], [-27, -9, 11], [0, 1, -1]])
    Phi = exp(-2*x) * Matrix([
        [2, 6*x - 2, 18*x**2 - 12*x - 2],
        [-3, -9*x + 9, -27*x**2 + 54*x],
        [3, 9*x, 27*x**2]
    ])
    verify_fundamental_matrix(A, Phi, x)
