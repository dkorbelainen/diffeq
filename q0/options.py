import sympy

# Declare x as a symbolic variable
x = sympy.symbols('x')

# Given parameter values
alpha0 = 2
beta0 = -3
gamma0 = 3

# Calculate K and L based on the given parameters
K = 9 * alpha0 + 2 * beta0 - 4 * gamma0
L = beta0 + gamma0

# --- Calculate psi1^(0) ---
psi1_0 = sympy.exp(-2 * x) * (2 * (K * x - (K + L) / 3) * sympy.log(sympy.Abs(x)) - 2 * L * x / 3 - 2 * K / 3 - alpha0 * x**(-1))

# --- Calculate psi2^(0) ---
psi2_0 = sympy.exp(-2 * x) * ((-3 * K * x + 3 * K + L) * sympy.log(sympy.Abs(x)) + L * x + 3 * K - beta0 * x**(-1))

# --- Calculate psi3^(0) ---
psi3_0 = sympy.exp(-2 * x) * ((3 * K * x - L) * sympy.log(sympy.Abs(x)) - L * x - gamma0 * x**(-1))

# Print the results
print("Expression for psi1^(0) after parameter substitution:")
print(psi1_0)
print("\nExpression for psi2^(0) after parameter substitution:")
print(psi2_0)
print("\nExpression for psi3^(0) after parameter substitution:")
print(psi3_0)