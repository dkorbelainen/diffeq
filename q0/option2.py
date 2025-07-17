import math

# Заданные параметры
alpha_0 = -2
beta_0 = 3
gamma_0 = -3

# Вычисление общего множителя (9*alpha_0 + 2*beta_0 - 4*gamma_0)
common_factor = 9 * alpha_0 + 2 * beta_0 - 4 * gamma_0

# Вычисление коэффициентов для psi_1^(0)
psi_1_coeff_x = 2 * common_factor
psi_1_coeff_const = 2 * (-(3 * alpha_0 + beta_0 - gamma_0))
psi_1_coeff_x_inv = 2 * (-alpha_0 / 2)
psi_1_coeff_const_term = 2 * (-(3 * alpha_0 + beta_0 - gamma_0))

# Вычисление коэффициентов для psi_2^(0)
psi_2_coeff_x = -3 * common_factor
psi_2_coeff_const = (27 * alpha_0 + 7 * beta_0 - 11 * gamma_0)
psi_2_coeff_x_inv = -beta_0
psi_2_coeff_const_term = (27 * alpha_0 + 7 * beta_0 - 11 * gamma_0)

# Вычисление коэффициентов для psi_3^(0)
psi_3_coeff_x = 3 * common_factor
psi_3_coeff_const = -(beta_0 + gamma_0)
psi_3_coeff_x_inv = -gamma_0
psi_3_coeff_const_term = -(beta_0 + gamma_0)

# Вывод упрощенных уравнений в виде строк
print(f"Подставляем параметры: alpha_0 = {alpha_0}, beta_0 = {beta_0}, gamma_0 = {gamma_0}")
print("---")
print("Уравнения с подставленными параметрами:")
print(f"psi_1^(0) = 2*e^(-2x) * ( ({psi_1_coeff_x}x + {psi_1_coeff_const})*ln|x| + {psi_1_coeff_x_inv}*x^-1 + {psi_1_coeff_const_term} )")
print(f"psi_2^(0) = e^(-2x) * ( ({psi_2_coeff_x}x + {psi_2_coeff_const})*ln|x| + {psi_2_coeff_x_inv}*x^-1 + {psi_2_coeff_const_term} )")
print(f"psi_3^(0) = e^(-2x) * ( ({psi_3_coeff_x}x + {psi_3_coeff_const})*ln|x| + {psi_3_coeff_x_inv}*x^-1 + {psi_3_coeff_const_term} )")