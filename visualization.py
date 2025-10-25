"""
Визуализация решений ЛНС
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, symbols, diff
import pandas as pd
from pathlib import Path


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


def analyze_function(x_vals, y_vals, label):
    """Анализ поведения функции: экстремумы, нули, статистика"""
    analysis = {
        'label': label,
        'min': np.min(y_vals),
        'max': np.max(y_vals),
        'mean': np.mean(y_vals),
        'std': np.std(y_vals)
    }

    # Поиск нулей (пересечения с осью x)
    zeros = []
    for i in range(len(y_vals) - 1):
        if y_vals[i] * y_vals[i+1] < 0:  # Смена знака
            zeros.append(x_vals[i])
    analysis['zeros'] = zeros[:3]  # Первые 3 нуля

    # Локальные экстремумы
    extrema_indices = []
    for i in range(1, len(y_vals) - 1):
        if (y_vals[i] > y_vals[i-1] and y_vals[i] > y_vals[i+1]) or \
           (y_vals[i] < y_vals[i-1] and y_vals[i] < y_vals[i+1]):
            extrema_indices.append(i)

    analysis['extrema_count'] = len(extrema_indices)
    if extrema_indices:
        analysis['first_extremum'] = (x_vals[extrema_indices[0]], y_vals[extrema_indices[0]])

    return analysis


def plot_solution(solution_dict, x_range=(0, 5), title="Решение ЛНС", save_path=None):
    """Улучшенный график компонент решения y₁(x), y₂(x), y₃(x) с анализом"""
    x_vals, y_vals = evaluate_solution(solution_dict, x_range)

    # Создаём фигуру с 4 подграфиками
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Заголовок
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    labels = ['y₁(x)', 'y₂(x)', 'y₃(x)']

    # Верхняя строка: индивидуальные графики с анализом
    analyses = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])

        # Анализ функции
        analysis = analyze_function(x_vals, y_vals[i], labels[i])
        analyses.append(analysis)

        # График
        ax.plot(x_vals, y_vals[i], color=colors[i], linewidth=2.5, label=labels[i], alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_ylabel(labels[i], fontsize=13, fontweight='bold', color=colors[i])
        ax.set_xlabel('x', fontsize=11)

        # Отмечаем нули функции
        for zero in analysis['zeros']:
            ax.axvline(x=zero, color=colors[i], linestyle=':', alpha=0.3, linewidth=1.5)

        # Добавляем текстовую аннотацию со статистикой
        stats_text = f"min: {analysis['min']:.3f}\nmax: {analysis['max']:.3f}"
        if analysis['zeros']:
            stats_text += f"\nнули: x≈{analysis['zeros'][0]:.2f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                fontsize=9)

        ax.legend(loc='upper left', fontsize=10)

    # Нижний левый: сводный график всех компонент
    ax_combined = fig.add_subplot(gs[1, :2])  # Занимает 2 колонки
    ax_combined.set_title('Сравнение всех компонент', fontsize=13, fontweight='bold')
    for i in range(3):
        ax_combined.plot(x_vals, y_vals[i], color=colors[i], linewidth=2.5,
                        label=labels[i], alpha=0.7)
    ax_combined.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax_combined.grid(True, alpha=0.3, linestyle=':')
    ax_combined.legend(loc='best', fontsize=11)
    ax_combined.set_xlabel('x', fontsize=12, fontweight='bold')
    ax_combined.set_ylabel('y', fontsize=12, fontweight='bold')

    # Нижний правый: таблица со статистикой
    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.axis('off')

    # Создаём таблицу с анализом
    table_data = []
    table_data.append(['Комп.', 'Min', 'Max', 'Mean', 'Std', 'Экстр.'])
    for analysis in analyses:
        row = [
            analysis['label'],
            f"{analysis['min']:.3f}",
            f"{analysis['max']:.3f}",
            f"{analysis['mean']:.3f}",
            f"{analysis['std']:.3f}",
            str(analysis['extrema_count'])
        ]
        table_data.append(row)

    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3)

    # Стилизация таблицы
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Заголовок
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(['#E7E6E6', '#F2F2F2'][i % 2])

    ax_table.set_title('Статистика', fontsize=12, fontweight='bold', pad=10)

    # Сохранение
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем график
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")

        # Сохраняем данные в CSV
        csv_path = save_path.with_suffix('.csv')
        df = pd.DataFrame({
            'x': x_vals,
            'y1': y_vals[0],
            'y2': y_vals[1],
            'y3': y_vals[2]
        })
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"✓ Данные сохранены: {csv_path}")

        # Сохраняем анализ в текстовый файл
        txt_path = save_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Диапазон: x ∈ [{x_range[0]}, {x_range[1]}]\n")
            f.write(f"Точек: {len(x_vals)}\n\n")

            for analysis in analyses:
                f.write(f"\n{analysis['label']}:\n")
                f.write(f"  Минимум: {analysis['min']:.6f}\n")
                f.write(f"  Максимум: {analysis['max']:.6f}\n")
                f.write(f"  Среднее: {analysis['mean']:.6f}\n")
                f.write(f"  Std Dev: {analysis['std']:.6f}\n")
                f.write(f"  Экстремумов: {analysis['extrema_count']}\n")
                if analysis['zeros']:
                    f.write(f"  Нули (первые 3): {[f'{z:.4f}' for z in analysis['zeros']]}\n")
                if 'first_extremum' in analysis:
                    x_ext, y_ext = analysis['first_extremum']
                    f.write(f"  Первый экстремум: x={x_ext:.4f}, y={y_ext:.4f}\n")

        print(f"✓ Анализ сохранен: {txt_path}")
    else:
        plt.show()

    plt.close()
    return fig


if __name__ == "__main__":
    # Пример использования
    from polynomial.solve import solve_polynomial_case
    from resonance.solve import solve_resonance_case

    print("Генерация улучшенных визуализаций...")

    # Создаём папку results если её нет
    Path("results").mkdir(exist_ok=True)

    # Полиномиальное решение
    psi_poly = solve_polynomial_case((-2, 2, -4), (9, -9, 27), (-1, 1, 1))
    plot_solution(psi_poly, x_range=(0, 3),
                 title="Полиномиальное решение",
                 save_path="results/polynomial_demo.png")

    # Резонансное решение
    psi_res = solve_resonance_case((-16, 2), (56, 1), (-8, 1))
    plot_solution(psi_res, x_range=(0, 2),
                 title="Резонансное решение",
                 save_path="results/resonance_demo.png")

    print("\nГотово! Проверьте папку results/")
