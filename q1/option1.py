# Задаем значения параметров
alpha1 = 2
beta1 = -11
gamma1 = 1

alpha2 = 0
beta2 = 0
gamma2 = 2

alpha3 = -6
beta3 = 36
gamma3 = -1

# Вычисляем коэффициенты для каждого уравнения
# psi_1
coeff_psi1_x2 = 2 * (-alpha1 + 2 * gamma1)
coeff_psi1_x1 = 2 * (13 * alpha1 + 2 * beta1 - 8 * gamma1 - alpha2 + 2 * gamma2)
coeff_psi1_c = -34 * alpha1 - 6 * beta1 + 18 * gamma1 + 13 * alpha2 + 2 * beta2 - 8 * gamma2 - 2 * alpha3 + 4 * gamma3
denom_psi1 = 8

# psi_2
coeff_psi2_x2 = 2 * (-27 * alpha1 - 4 * beta1 + 10 * gamma1)
coeff_psi2_x1 = 2 * (27 * alpha1 + 6 * beta1 - 8 * gamma1 - 27 * alpha2 - 4 * beta2 + 10 * gamma2)
coeff_psi2_c = -2 * beta1 - 6 * gamma1 + 27 * alpha2 + 6 * beta2 - 8 * gamma2 - 54 * alpha3 - 8 * beta3 + 20 * gamma3
denom_psi2 = 16

# psi_3
coeff_psi3_x2 = 2 * (-27 * alpha1 - 4 * beta1 + 18 * gamma1)
coeff_psi3_x1 = 2 * (81 * alpha1 + 14 * beta1 - 44 * gamma1 - 27 * alpha2 - 4 * beta2 + 18 * gamma2)
coeff_psi3_c = -162 * alpha1 - 30 * beta1 + 82 * gamma1 + 81 * alpha2 + 14 * beta2 - 44 * gamma2 - 54 * alpha3 - 8 * beta3 + 36 * gamma3
denom_psi3 = 16

# Выводим упрощенные выражения
print(f"psi_1^(1) = ({coeff_psi1_x2} * x^2 + {coeff_psi1_x1} * x + {coeff_psi1_c}) / {denom_psi1}")
print(f"psi_2^(1) = ({coeff_psi2_x2} * x^2 + {coeff_psi2_x1} * x + {coeff_psi2_c}) / {denom_psi2}")
print(f"psi_3^(1) = ({coeff_psi3_x2} * x^2 + {coeff_psi3_x1} * x + {coeff_psi3_c}) / {denom_psi3}")

# Выводим окончательно упрощенные выражения, убирая нулевые коэффициенты
print("\n--- Упрощенный вид ---")
psi1_result = f"({coeff_psi1_x2 * 1/denom_psi1} * x^2 + {coeff_psi1_x1 * 1/denom_psi1} * x + {coeff_psi1_c * 1/denom_psi1})"
psi2_result = f"({coeff_psi2_x2 * 1/denom_psi2} * x^2 + {coeff_psi2_x1 * 1/denom_psi2} * x + {coeff_psi2_c * 1/denom_psi2})"
psi3_result = f"({coeff_psi3_x2 * 1/denom_psi3} * x^2 + {coeff_psi3_x1 * 1/denom_psi3} * x + {coeff_psi3_c * 1/denom_psi3})"

# Очистка вывода от лишних нулей и скобок для более красивого представления
psi1_final = psi1_result.replace("0.0 * x^2", "").replace("0.0 * x", "").replace("+ ", "").strip("() ")
psi2_final = psi2_result.replace("0.0 * x^2", "").replace("0.0 * x", "").replace("+ ", "").strip("() ")
psi3_final = psi3_result.replace("0.0 * x^2", "").replace("0.0 * x", "").replace("+ ", "").strip("() ")
print(f"psi_1^(1) = {psi1_final}")
print(f"psi_2^(1) = {psi2_final}")
print(f"psi_3^(1) = {psi3_final}")