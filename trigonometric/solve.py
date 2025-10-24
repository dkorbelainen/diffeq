"""Тригонометрия (ax+b)sin(x) — метод неопределенных коэффициентов"""

from sympy import Symbol, sin, cos, simplify


def solve_trigonometric_case(alpha, beta, gamma):
    """Решение для q(x) = [(α₆x+α₇)sin(x), (β₆x+β₇)sin(x), (γ₆x+γ₇)sin(x)]ᵀ"""
    x = Symbol('x')
    alpha_6, alpha_7 = alpha
    beta_6, beta_7 = beta
    gamma_6, gamma_7 = gamma

    # Коэффициенты получены методом неопределенных коэффициентов
    psi1 = (
        sin(x) * (-573*alpha_6 - 124*beta_6 + 208*gamma_6 + 520*alpha_7 + 110*beta_7 - 70*gamma_7 + 10*x*(52*alpha_6 + 11*beta_6 - 7*gamma_6)) -
        cos(x) * (536*alpha_6 + 68*beta_6 - 356*gamma_6 - 265*alpha_7 - 20*beta_7 + 240*gamma_7 - 5*x*(53*alpha_6 + 4*beta_6 - 48*gamma_6))
    ) / 625

    psi2 = (
        sin(x) * (1107*alpha_6 + 191*beta_6 - 472*gamma_6 - 1755*alpha_7 - 215*beta_7 + 705*gamma_7 - 5*x*(351*alpha_6 + 43*beta_6 - 141*gamma_6)) -
        cos(x) * (1026*alpha_6 + 238*beta_6 - 346*gamma_6 - 1215*alpha_7 - 245*beta_7 + 440*gamma_7 - 5*x*(243*alpha_6 + 49*beta_6 - 88*gamma_6))
    ) / 625

    psi3 = (
        sin(x) * (-567*alpha_6 - 146*beta_6 + 157*gamma_6 - 270*alpha_7 + 15*beta_7 + 445*gamma_7 - 5*x*(54*alpha_6 - 3*beta_6 - 89*gamma_6)) -
        cos(x) * (1944*alpha_6 + 322*beta_6 - 1074*gamma_6 - 1485*alpha_7 - 230*beta_7 + 885*gamma_7 - 5*x*(297*alpha_6 + 46*beta_6 - 177*gamma_6))
    ) / 625

    return {
        'psi1': simplify(psi1),
        'psi2': simplify(psi2),
        'psi3': simplify(psi3)
    }


if __name__ == "__main__":
    result = solve_trigonometric_case((48, 4), (-168, -32), (39, 0))
    print("\nРешение:")
    for i in range(1, 4):
        print(f"  ψ_{i}(x) = {result[f'psi{i}']}")
