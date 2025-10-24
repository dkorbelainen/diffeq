"""Особая точка x⁻²e⁻²ˣ — метод вариации постоянных"""

from sympy import symbols, Matrix, exp, log, integrate, simplify, Abs


def solve_special_case(alpha0, beta0, gamma0, verbose=True):
    """Решение для q(x) = x⁻²e⁻²ˣ·[α₀, β₀, γ₀]ᵀ"""
    x = symbols('x', real=True, positive=True)

    Phi = Matrix([
        [2, 6*x - 2, 18*x**2 - 12*x - 2],
        [-3, 9 - 9*x, 54*x - 27*x**2],
        [3, 9*x, 27*x**2]
    ])

    q0_reduced = Matrix([alpha0/x**2, beta0/x**2, gamma0/x**2])
    c_prime = Phi.inv() * q0_reduced
    c = Matrix([integrate(expr, x) for expr in c_prime])

    K = 9*alpha0 + 2*beta0 - 4*gamma0
    L = beta0 + gamma0

    psi1 = exp(-2*x) * (2*(K*x - (K + L)/3)*log(Abs(x)) - 2*L*x/3 - 2*K/3 - alpha0/x)
    psi2 = exp(-2*x) * ((-3*K*x + 3*K + L)*log(Abs(x)) + L*x + 3*K - beta0/x)
    psi3 = exp(-2*x) * ((3*K*x - L)*log(Abs(x)) - L*x - gamma0/x)

    return {
        'psi1': simplify(psi1),
        'psi2': simplify(psi2),
        'psi3': simplify(psi3),
        'K': K,
        'L': L
    }


if __name__ == "__main__":
    result = solve_special_case(2, -3, 3)
    print(f"\nКонстанты: K={result['K']}, L={result['L']}")
    print("\nРешение:")
    for i in range(1, 4):
        print(f"  ψ_{i}(x) = {result[f'psi{i}']}")
