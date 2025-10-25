"""Численное решение и сравнение с аналитическим"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from visualization import evaluate_solution


def solve_numerically(A, q_func, y0, x_range=(0, 5), n_points=100):
    """Численное решение y' = Ay + q(x) методом Рунге-Кутты"""
    def system(x, y):
        return A @ y + q_func(x)

    solution = solve_ivp(
        system, x_range, y0,
        method='RK45', dense_output=True,
        rtol=1e-10, atol=1e-12
    )

    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    y_vals = solution.sol(x_vals)

    return x_vals, y_vals


def analyze_solution(solution_dict, A_numpy, q_func, y0, x_range=(0, 5),
                     solution_name="Решение", save_prefix=None):
    """Анализ: численное решение, сравнение, метрики"""
    print(f"\n{'='*70}")
    print(f"  ЧИСЛЕННЫЙ АНАЛИЗ: {solution_name}")
    print(f"{'='*70}")

    # Численное решение
    print("\nРешение методом Рунге-Кутты 4-5...")
    x_vals, y_numerical = solve_numerically(A_numpy, q_func, y0, x_range, n_points=100)

    # Аналитическое решение
    _, y_analytical = evaluate_solution(solution_dict, x_range, n_points=len(x_vals))

    # Погрешность
    abs_error = np.abs(y_analytical - y_numerical)

    print("\nПогрешность:")
    for i in range(3):
        max_err = np.max(abs_error[i])
        mean_err = np.mean(abs_error[i])
        print(f"  y_{i+1}: max={max_err:.2e}, mean={mean_err:.2e}")

    # График сравнения
    if save_prefix:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("Аналитическое и численное решения",
                     fontsize=16, fontweight='bold')

        labels = ['y₁(x)', 'y₂(x)', 'y₃(x)']
        colors_ana = ['#2E86AB', '#A23B72', '#F18F01']
        colors_num = ['#06D6A0', '#E63946', '#FFD60A']

        for i in range(3):
            # Левая колонка: сравнение
            ax = axes[i, 0]
            ax.plot(x_vals, y_analytical[i], label=f'{labels[i]} (аналитическое)',
                    color=colors_ana[i], linewidth=2, linestyle='-')
            ax.plot(x_vals, y_numerical[i], label=f'{labels[i]} (численное)',
                    color=colors_num[i], linewidth=1.5, linestyle='--', alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(labels[i], fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)

            # Правая колонка: погрешность
            ax = axes[i, 1]
            error = abs_error[i]
            ax.semilogy(x_vals, error, color='red', linewidth=2)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylabel(f'|Погрешность| {labels[i]}', fontsize=11, fontweight='bold')

            max_err = np.max(error)
            mean_err = np.mean(error)
            ax.text(0.02, 0.98, f'max={max_err:.2e}\nmean={mean_err:.2e}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontsize=9)

        axes[-1, 0].set_xlabel('x', fontsize=12)
        axes[-1, 1].set_xlabel('x', fontsize=12)
        plt.tight_layout()

        plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')
        print(f"\n✓ График сохранен: {save_prefix}.png")

        # Сохранение CSV
        data = {
            'x': x_vals,
            'y1_analytical': y_analytical[0],
            'y1_numerical': y_numerical[0],
            'y1_error': abs_error[0],
            'y2_analytical': y_analytical[1],
            'y2_numerical': y_numerical[1],
            'y2_error': abs_error[1],
            'y3_analytical': y_analytical[2],
            'y3_numerical': y_numerical[2],
            'y3_error': abs_error[2],
        }

        df_data = pd.DataFrame(data)
        df_data.to_csv(f"{save_prefix}.csv", index=False)
        print(f"✓ Данные сохранены: {save_prefix}.csv")

    print(f"{'='*70}\n")

    return y_numerical, abs_error


if __name__ == "__main__":
    print("Демонстрация численного анализа")
    print("="*70)

    # Пример: полиномиальное решение
    from polynomial.solve import solve_polynomial_case

    # Аналитическое решение
    psi = solve_polynomial_case((-2, 2, -4), (9, -9, 27), (-1, 1, 1))

    # Матрица A
    A = np.array([[4, 2, -2],
                  [-27, -9, 11],
                  [0, 1, -1]], dtype=float)

    # Неоднородность q(x) = [-2x² + 2x - 4, 9x² - 9x + 27, -x² + x + 1]ᵀ
    def q_poly(x):
        return np.array([
            -2*x**2 + 2*x - 4,
            9*x**2 - 9*x + 27,
            -x**2 + x + 1
        ])

    # Начальные условия (вычисляем из аналитического решения при x=0)
    y0 = np.array([psi['psi1'][2], psi['psi2'][2], psi['psi3'][2]])

    # Анализ
    analyze_solution(psi, A, q_poly, y0, x_range=(0, 3),
                    solution_name="Полиномиальное", save_prefix="poly")
