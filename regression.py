"""
Регрессия для обратных задач: восстановление A и q(x) по решению y(x)
"""

import numpy as np
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from sympy import symbols, simplify
import matplotlib.pyplot as plt
from pathlib import Path


class ParameterReconstructor:
    """Восстановление параметров системы y' = Ay + q(x) по данным решения"""

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.A_reconstructed = None
        self.q_formulas = None

    def solve(self, x_data, y_data, A_true=None):
        """
        Полное решение обратной задачи

        Args:
            x_data: значения x, shape (n_samples,)
            y_data: решение y(x), shape (n_samples, n_dim)
            A_true: истинная матрица A (опционально, для валидации)

        Returns:
            dict с результатами
        """
        print("\n" + "="*60)
        print("  РЕГРЕССИЯ: ВОССТАНОВЛЕНИЕ ПАРАМЕТРОВ")
        print("="*60)

        # 1. Численная производная
        y_prime = np.gradient(y_data, x_data, axis=0)

        # 2. Восстановление матрицы A (предполагая q≈0)
        print("\n[1] Восстановление матрицы A (Ridge регрессия)...")
        self.A_reconstructed = self._reconstruct_A(y_data, y_prime)

        # 3. Восстановление q(x) из невязки
        print("\n[2] Восстановление q(x) (символическая регрессия)...")
        q_residual = y_prime - (self.A_reconstructed @ y_data.T).T
        self.q_formulas = self._reconstruct_q(x_data, q_residual)

        # 4. Уточнение A с учетом q
        print("\n[3] Уточнение A с учетом найденного q(x)...")
        self.A_reconstructed = self._reconstruct_A(y_data, y_prime, q_residual)

        # Метрики
        result = {
            'A': self.A_reconstructed,
            'q_formulas': self.q_formulas,
        }

        if A_true is not None:
            error = np.linalg.norm(self.A_reconstructed - A_true, 'fro')
            rel_error = error / np.linalg.norm(A_true, 'fro')
            result['error'] = rel_error
            print(f"\nОшибка восстановления A: {rel_error:.1%}")

        print("\nВосстановленная матрица A:")
        print(self.A_reconstructed)
        print("\nВосстановленные формулы q(x):")
        for key, val in self.q_formulas.items():
            print(f"  {key}(x) = {val['formula']}")
            print(f"       Тип: {val['type']}, R² = {val['r2']:.3f}")

        print("="*60)
        return result

    def _reconstruct_A(self, y_data, y_prime, q_data=None):
        """Восстановление матрицы A через Ridge регрессию"""
        n_dim = y_data.shape[1]
        A = np.zeros((n_dim, n_dim))

        if q_data is None:
            q_data = np.zeros_like(y_data)

        for i in range(n_dim):
            target = y_prime[:, i] - q_data[:, i]
            model = Ridge(alpha=self.alpha)
            model.fit(y_data, target)
            A[i, :] = model.coef_

        return A

    def _reconstruct_q(self, x_data, q_data, max_degree=3):
        """Символическая регрессия для q(x)"""
        n_dim = q_data.shape[1]
        formulas = {}

        for i in range(n_dim):
            # Пробуем разные базисы
            results = []

            # Полином
            poly_result = self._fit_polynomial(x_data, q_data[:, i], max_degree)
            results.append(('poly', poly_result))

            # Экспонента
            exp_result = self._fit_exponential(x_data, q_data[:, i])
            results.append(('exp', exp_result))

            # Тригонометрия
            trig_result = self._fit_trigonometric(x_data, q_data[:, i])
            results.append(('trig', trig_result))

            # Выбираем лучший с учетом приоритетов
            # Если exp или trig всего на 10% хуже полинома по R², выбираем их
            best_r2 = max(r[1]['r2'] for r in results)
            threshold = 0.10  # 10% разница

            # Приоритет: экспонента > тригонометрия > полином (при близких R²)
            selected = None
            for func_type in ['exp', 'trig', 'poly']:
                candidate = next((r for r in results if r[0] == func_type), None)
                if candidate and candidate[1]['r2'] >= best_r2 - threshold:
                    selected = candidate
                    break

            if selected is None:
                selected = max(results, key=lambda x: x[1]['r2'])

            formulas[f'q{i+1}'] = selected[1]

        return formulas

    def _fit_polynomial(self, x, y, degree):
        """Подбор полинома"""
        from sklearn.metrics import r2_score

        best_r2 = -np.inf
        best_params = None
        best_deg = 0

        for d in range(degree + 1):
            params = np.polyfit(x, y, d)
            y_pred = np.polyval(params, x)
            r2 = r2_score(y, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_params = params
                best_deg = d

        # Формула с округлением
        x_sym = symbols('x')
        rounded_params = [round(p, 2) for p in best_params]
        formula = sum(p * x_sym**i for i, p in enumerate(reversed(rounded_params)))

        return {
            'formula': str(simplify(formula)),
            'r2': best_r2,
            'type': 'polynomial',
            'params': best_params
        }

    def _fit_exponential(self, x, y):
        """Подбор a*exp(b*x) + c"""
        from sklearn.metrics import r2_score

        def loss(params):
            a, b, c = params
            try:
                y_pred = a * np.exp(b * x) + c
                return np.mean((y - y_pred)**2)
            except:
                return 1e10

        best_r2 = -np.inf
        best_result = None

        # Пробуем несколько начальных приближений
        initial_guesses = [
            [1.0, -1.0, 0.0],   # убывающая экспонента
            [1.0, 1.0, 0.0],    # растущая экспонента
            [-1.0, -1.0, 0.0],  # отрицательная убывающая
            [0.5, -0.5, 0.0],   # слабая
        ]

        for init in initial_guesses:
            try:
                result = minimize(loss, init, method='L-BFGS-B',
                                bounds=[(-100, 100), (-10, 10), (-100, 100)])
                a, b, c = result.x
                y_pred = a * np.exp(b * x) + c
                r2 = r2_score(y, y_pred)

                if r2 > best_r2:
                    best_r2 = r2
                    best_result = (a, b, c)
            except:
                continue

        if best_result is None:
            return {'formula': '0', 'r2': -np.inf, 'type': 'exponential', 'params': [0, 0, 0]}

        a, b, c = best_result
        return {
            'formula': f"{a:.2f}*exp({b:.2f}*x) + {c:.2f}",
            'r2': best_r2,
            'type': 'exponential',
            'params': [a, b, c]
        }

    def _fit_trigonometric(self, x, y):
        """Подбор a*sin(w*x) + b*cos(w*x) + c"""
        from sklearn.metrics import r2_score

        def loss(params):
            a, w, b, c = params
            y_pred = a * np.sin(w * x) + b * np.cos(w * x) + c
            return np.mean((y - y_pred)**2)

        best_r2 = -np.inf
        best_result = None

        # Пробуем разные частоты
        frequencies = [1.0, 2.0, np.pi, 2*np.pi, 5.0, 10.0]

        for w_init in frequencies:
            try:
                result = minimize(loss, [1.0, w_init, 0.0, 0.0], method='L-BFGS-B',
                                bounds=[(-100, 100), (0.1, 20), (-100, 100), (-100, 100)])
                a, w, b, c = result.x
                y_pred = a * np.sin(w * x) + b * np.cos(w * x) + c
                r2 = r2_score(y, y_pred)

                if r2 > best_r2:
                    best_r2 = r2
                    best_result = (a, w, b, c)
            except:
                continue

        if best_result is None:
            return {'formula': '0', 'r2': -np.inf, 'type': 'trigonometric', 'params': [0, 0, 0, 0]}

        a, w, b, c = best_result

        # Упрощаем формулу если амплитуда мала
        if abs(c) < 0.01:
            formula = f"{a:.2f}*sin({w:.2f}*x) + {b:.2f}*cos({w:.2f}*x)"
        else:
            formula = f"{a:.2f}*sin({w:.2f}*x) + {b:.2f}*cos({w:.2f}*x) + {c:.2f}"

        return {
            'formula': formula,
            'r2': best_r2,
            'type': 'trigonometric',
            'params': [a, w, b, c]
        }

    def plot_results(self, x_data, y_data, save_dir='results'):
        """Визуализация результатов регрессии"""
        if self.A_reconstructed is None:
            print("Ошибка: Сначала выполните solve()")
            return

        Path(save_dir).mkdir(exist_ok=True)

        # Вычисляем данные для графиков
        y_prime = np.gradient(y_data, x_data, axis=0)
        q_residual = y_prime - (self.A_reconstructed @ y_data.T).T

        n_dim = y_data.shape[1]

        # Цветовая схема для разных типов
        type_colors = {
            'polynomial': '#e74c3c',      # красный
            'exponential': '#2ecc71',     # зеленый
            'trigonometric': '#9b59b6'    # фиолетовый
        }

        # Создаём график 2x3: верхний ряд - графики q(x), нижний - матрица A и таблица
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Верхний ряд - графики q(x)
        for i in range(n_dim):
            ax = fig.add_subplot(gs[0, i])

            formula_info = self.q_formulas[f'q{i+1}']
            func_type = formula_info['type']
            color = type_colors.get(func_type, '#34495e')

            # Невязка (данные)
            ax.scatter(x_data, q_residual[:, i], alpha=0.4, s=10,
                      color='steelblue', label='Невязка из данных')

            # Вычисляем предсказание по формуле
            if func_type == 'polynomial':
                y_pred = np.polyval(formula_info['params'], x_data)
            elif func_type == 'exponential':
                a, b, c = formula_info['params']
                y_pred = a * np.exp(b * x_data) + c
            elif func_type == 'trigonometric':
                a, w, b, c = formula_info['params']
                y_pred = a * np.sin(w * x_data) + b * np.cos(w * x_data) + c
            else:
                y_pred = np.zeros_like(x_data)

            ax.plot(x_data, y_pred, '-', linewidth=2.5, color=color,
                   label=f'Регрессия (R²={formula_info["r2"]:.3f})')

            ax.set_xlabel('x', fontsize=11)
            ax.set_ylabel(f'q_{i+1}(x)', fontsize=11)
            ax.set_title(f'Компонента q_{i+1}(x)\nТип: {func_type}',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

        # Нижний ряд слева - матрица A
        ax_matrix = fig.add_subplot(gs[1, :2])

        im = ax_matrix.imshow(self.A_reconstructed, cmap='RdBu_r', aspect='auto')
        ax_matrix.set_title('Восстановленная матрица A (цветовая карта)', fontsize=12, fontweight='bold')
        ax_matrix.set_xlabel('Столбец')
        ax_matrix.set_ylabel('Строка')

        # Добавляем значения в ячейки
        for i in range(self.A_reconstructed.shape[0]):
            for j in range(self.A_reconstructed.shape[1]):
                text = ax_matrix.text(j, i, f'{self.A_reconstructed[i, j]:.2f}',
                                    ha='center', va='center', color='black', fontsize=9)

        plt.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

        # Нижний ряд справа - таблица с формулами
        ax_table = fig.add_subplot(gs[1, 2])
        ax_table.axis('off')

        table_data = []
        for i in range(n_dim):
            formula_info = self.q_formulas[f'q{i+1}']

            table_data.append([
                f'q_{i+1}(x)',
                formula_info['type'],
                f'{formula_info["r2"]:.3f}'
            ])

        table = ax_table.table(cellText=table_data,
                              colLabels=['Функция', 'Тип', 'R²'],
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0.45, 1, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3.5)

        # Стилизация заголовка таблицы
        for i in range(3):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Заголовок выше
        ax_table.text(0.5, 0.92, 'Результаты регрессии', transform=ax_table.transAxes,
                     fontsize=12, fontweight='bold', ha='center', va='top')

        # Добавляем формулы текстом ниже таблицы
        formulas_text = "Формулы:\n"
        for i in range(n_dim):
            formula_info = self.q_formulas[f'q{i+1}']
            formulas_text += f"q_{i+1}(x) = {formula_info['formula']}\n"

        ax_table.text(0.5, 0.15, formulas_text, transform=ax_table.transAxes,
                     fontsize=9, verticalalignment='top', horizontalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Регрессионный анализ: восстановление параметров системы y\' = Ay + q(x)',
                    fontsize=14, fontweight='bold', y=0.98)

        save_path = Path(save_dir) / 'images' / 'regression_analysis.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Сохранено: {save_path}")
        plt.close()


def demo():
    """Быстрая демонстрация"""
    from scipy.integrate import odeint

    print("\n" + "="*60)
    print("  DEMO: Регрессионный анализ")
    print("="*60)

    A_true = np.array([
        [1, 0.5, 0],
        [0, -1, 0.5],
        [0, 0, -0.5]
    ], dtype=float)

    def q_func(t):
        return np.array([
            0.8 * np.exp(-t),             # экспонента
            0.5 * np.sin(2*np.pi*t),      # тригонометрия
            0.2*t**2 - 0.3*t + 0.1        # полином
        ])

    def system(y, t):
        return A_true @ y + q_func(t)

    # Данные
    x_data = np.linspace(0, 2.5, 200)
    y0 = np.array([1.0, 0.5, 0.2])
    y_data = odeint(system, y0, x_data)

    print(f"\nИстинная система:")
    print(f"   q₁(x) = 0.8·e^(-x)          (экспонента)")
    print(f"   q₂(x) = 0.5·sin(2πx)        (тригонометрия)")
    print(f"   q₃(x) = 0.2x² - 0.3x + 0.1  (полином)")
    print(f"   Матрица A известна для валидации")

    # Регрессионный анализ
    solver = ParameterReconstructor(alpha=0.01)
    result = solver.solve(x_data, y_data, A_true=A_true)

    # Визуализация
    solver.plot_results(x_data, y_data)

    return result


if __name__ == '__main__':
    demo()
