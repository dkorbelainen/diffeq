"""
Визуализация решений ЛНС
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, symbols


def evaluate_solution(solution_dict, x_range=(0, 5), n_points=500):
    """Преобразует символьное решение в численные значения"""
    x = symbols('x')
    x_vals = np.linspace(x_range[0], x_range[1], n_points)

    y_vals = []
    for i in range(1, 4):
        key = f'psi{i}'
        if key in solution_dict:
            psi = solution_dict[key]

            if isinstance(psi, tuple):
                # Полином: (a2, a1, a0)
                a2, a1, a0 = psi
                vals = a2 * x_vals**2 + a1 * x_vals + a0
            else:
                # Символьное выражение
                func = lambdify(x, psi, modules=['numpy'])
                try:
                    vals = func(x_vals)
                    if np.iscomplexobj(vals):
                        vals = np.real(vals)
                    if np.isscalar(vals):
                        vals = np.full_like(x_vals, vals, dtype=float)
                except Exception as e:
                    print(f"Ошибка при вычислении {key}: {e}")
                    vals = np.zeros_like(x_vals)

            y_vals.append(vals)
        else:
            y_vals.append(np.zeros_like(x_vals))

    return x_vals, np.array(y_vals)


def plot_solution(solution_dict, x_range=(0, 5), title="Решение ЛНС", save_path=None):
    """График компонент решения y₁(x), y₂(x), y₃(x)"""
    x_vals, y_vals = evaluate_solution(solution_dict, x_range)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    labels = ['y₁(x)', 'y₂(x)', 'y₃(x)']

    for i, (ax, color, label) in enumerate(zip(axes, colors, labels)):
        ax.plot(x_vals, y_vals[i], color=color, linewidth=2, label=label)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('x', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    # Пример использования
    from polynomial.solve import solve_polynomial_case
    from resonance.solve import solve_resonance_case

    print("Генерация визуализаций...")

    # Полиномиальное решение
    psi_poly = solve_polynomial_case((-2, 2, -4), (9, -9, 27), (-1, 1, 1))
    plot_solution(psi_poly, x_range=(0, 3), title="Полиномиальное решение")

    # Резонансное решение
    psi_res = solve_resonance_case((-16, 2), (56, 1), (-8, 1))
    plot_solution(psi_res, x_range=(0, 2), title="Резонансное решение")
