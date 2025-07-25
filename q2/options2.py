import sympy

# Define the symbolic variable x
x = sympy.Symbol('x')

# Define the coefficients
alpha_4 = 16
beta_4 = -72
gamma_4 = 0

alpha_5 = -8
beta_5 = 24
gamma_5 = 0

# Substitute the coefficients into the equations
# Equation for psi_1^(2)
psi1_numerator = -6 * (9 * alpha_4 + 2 * beta_4 - 8 * gamma_4) * x**4 + \
                 24 * (3 * alpha_4 + beta_4 - 3 * gamma_4 - 9 * alpha_5 - 2 * beta_5 + 4 * gamma_5) * x**3 + \
                 36 * (beta_4 - gamma_4 - 12 * alpha_5 - 4 * beta_5 + 8 * gamma_5) * x**2 + \
                 6 * (beta_4 + gamma_4 - 24 * alpha_5 - 8 * beta_5 + 16 * gamma_5) * x + \
                 (beta_4 + gamma_4 - 42 * alpha_5 - 8 * beta_5 + 44 * gamma_5)

psi1_2 = sympy.exp(-2 * x) * psi1_numerator / 72

# Equation for psi_2^(2)
psi2_numerator = 3 * (9 * alpha_4 + 2 * beta_4 - 8 * gamma_4) * x**4 - \
                 4 * (27 * alpha_4 + 7 * beta_4 - 25 * gamma_4 - 27 * alpha_5 - 6 * beta_5 + 12 * gamma_5) * x**3 + \
                 12 * (beta_4 - gamma_4 - 27 * alpha_5 - 6 * beta_5 + 12 * gamma_5) * x**2 - \
                 24 * gamma_4 * x - 24 * gamma_5

psi2_2 = sympy.exp(-2 * x) * psi2_numerator / 24

# Equation for psi_3^(2)
psi3_numerator = -3 * (9 * alpha_4 + 2 * beta_4 - 8 * gamma_4) * x**4 + \
                 4 * (beta_4 - gamma_4 - 27 * alpha_5 - 6 * beta_5 + 12 * gamma_5) * x**3

psi3_2 = sympy.exp(-2 * x) * psi3_numerator / 24

# Simplify the expressions
psi1_2_simplified = sympy.simplify(psi1_2)
psi2_2_simplified = sympy.simplify(psi2_2)
psi3_2_simplified = sympy.simplify(psi3_2)

# Print the results
print("Simplified expressions:")
print(f"psi_1^(2) = {psi1_2_simplified}")
print(f"psi_2^(2) = {psi2_2_simplified}")
print(f"psi_3^(2) = {psi3_2_simplified}")