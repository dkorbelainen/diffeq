"""
Полином ax²+bx+c — метод неопределенных коэффициентов
"""


def solve_polynomial_case(alpha, beta, gamma):
    """Решение для q(x) = [α₁x²+α₂x+α₃, β₁x²+β₂x+β₃, γ₁x²+γ₂x+γ₃]ᵀ"""
    alpha1, alpha2, alpha3 = alpha
    beta1, beta2, beta3 = beta
    gamma1, gamma2, gamma3 = gamma

    a2 = (-alpha1 + 2*gamma1) / 4
    b2 = (-27*alpha1 - 4*beta1 + 10*gamma1) / 8
    c2 = (-27*alpha1 - 4*beta1 + 18*gamma1) / 8

    a1 = (13*alpha1 + 2*beta1 - 8*gamma1 - alpha2 + 2*gamma2) / 4
    b1 = (27*alpha1 + 6*beta1 - 8*gamma1 - 27*alpha2 - 4*beta2 + 10*gamma2) / 8
    c1 = (81*alpha1 + 14*beta1 - 44*gamma1 - 27*alpha2 - 4*beta2 + 18*gamma2) / 8

    a0 = (-34*alpha1 - 6*beta1 + 18*gamma1 + 13*alpha2 + 2*beta2 - 8*gamma2 - 2*alpha3 + 4*gamma3) / 8
    b0 = (-2*beta1 - 6*gamma1 + 27*alpha2 + 6*beta2 - 8*gamma2 - 54*alpha3 - 8*beta3 + 20*gamma3) / 16
    c0 = (-162*alpha1 - 30*beta1 + 82*gamma1 + 81*alpha2 + 14*beta2 - 44*gamma2 - 54*alpha3 - 8*beta3 + 36*gamma3) / 16

    return {
        'psi1': (a2, a1, a0),
        'psi2': (b2, b1, b0),
        'psi3': (c2, c1, c0)
    }


def format_polynomial(coeffs):
    """Красивый вывод полинома"""
    a2, a1, a0 = coeffs
    terms = []
    if abs(a2) > 1e-10:
        terms.append(f"{a2:.4g}x²" if abs(abs(a2) - 1) > 1e-10 else ("x²" if a2 > 0 else "-x²"))
    if abs(a1) > 1e-10:
        sign = " + " if a1 > 0 and terms else ("" if a1 > 0 else "-")
        val = abs(a1)
        terms.append(f"{sign}{val:.4g}x" if abs(val - 1) > 1e-10 else f"{sign}x")
    if abs(a0) > 1e-10:
        sign = " + " if a0 > 0 and terms else ("" if a0 > 0 else "-")
        val = abs(a0)
        terms.append(f"{sign}{val:.4g}")
    return "".join(terms) if terms else "0"


if __name__ == "__main__":
    result = solve_polynomial_case((-2, 2, -4), (9, -9, 27), (-1, 1, 1))
    print("\nРешение:")
    for i in range(1, 4):
        print(f"  ψ_{i}(x) = {format_polynomial(result[f'psi{i}'])}")
