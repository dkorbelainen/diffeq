"""Резонанс e⁻²ˣ·(ax+b) — λ=-2 совпадает с собственным числом"""

from sympy import Symbol, exp, simplify


def solve_resonance_case(alpha, beta, gamma):
    """Решение для q(x) = e⁻²ˣ·[α₄x+α₅, β₄x+β₅, γ₄x+γ₅]ᵀ"""
    x = Symbol('x')
    alpha_4, alpha_5 = alpha
    beta_4, beta_5 = beta
    gamma_4, gamma_5 = gamma

    a3 = (6*(3*alpha_4 + beta_4 - gamma_4 - 9*alpha_5 - 2*beta_5 + 4*gamma_5)) / 18
    a2 = (3*(3*alpha_4 - 2*gamma_4 + 18*alpha_5 + 4*beta_5 - 8*gamma_5)) / 18
    a1 = (2*(2*gamma_4 + 9*alpha_5 + 2*beta_5 - 4*gamma_5)) / 18
    a0 = (2*(gamma_4 + beta_5 + 7*gamma_5)) / 18
    psi1 = exp(-2*x) * (a3*x**3 + a2*x**2 + a1*x + a0)

    b3 = ((-27*alpha_4 - 7*beta_4 + 11*gamma_4 + 27*alpha_5 + 6*beta_5 - 12*gamma_5)) / 6
    b2 = (3*(beta_4 + gamma_4 - 27*alpha_5 - 6*beta_5 + 12*gamma_5)) / 6
    b1 = -gamma_4
    b0 = -gamma_5
    psi2 = exp(-2*x) * (b3*x**3 + b2*x**2 + b1*x + b0)

    c3 = (beta_4 + gamma_4 - 27*alpha_5 - 6*beta_5 + 12*gamma_5) / 6
    psi3 = exp(-2*x) * (c3*x**3)

    return {
        'psi1': simplify(psi1),
        'psi2': simplify(psi2),
        'psi3': simplify(psi3)
    }


if __name__ == "__main__":
    result = solve_resonance_case((-16, 2), (56, 1), (-8, 1))
    print("\nРешение:")
    for i in range(1, 4):
        print(f"  ψ_{i}(x) = {result[f'psi{i}']}")
