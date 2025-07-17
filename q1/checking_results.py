from sympy import symbols, Matrix, diff, simplify, expand, latex

# Define symbolic variables
x = symbols('x')
alpha1, alpha2, alpha3, beta1, beta2, beta3, gamma1, gamma2, gamma3 = symbols('alpha1 alpha2 alpha3 beta1 beta2 beta3 gamma1 gamma2 gamma3')

# Define coefficients in the transformed form
a2 = (-alpha1 + 2*gamma1) / 4
b2 = (-27*alpha1 - 4*beta1 + 10*gamma1) / 8
c2 = (-27*alpha1 - 4*beta1 + 18*gamma1) / 8

a1 = (13*alpha1 + 2*beta1 - 8*gamma1 - alpha2 + 2*gamma2) / 4
b1 = (27*alpha1 + 6*beta1 - 8*gamma1 - 27*alpha2 - 4*beta2 + 10*gamma2) / 8
c1 = (81*alpha1 + 14*beta1 - 44*gamma1 - 27*alpha2 - 4*beta2 + 18*gamma2) / 8

a0 = (-34*alpha1 - 6*beta1 + 18*gamma1 + 13*alpha2 + 2*beta2 - 8*gamma2 - 2*alpha3 + 4*gamma3) / 8
b0 = (-2*beta1 - 6*gamma1 + 27*alpha2 + 6*beta2 - 8*gamma2 - 54*alpha3 - 8*beta3 + 20*gamma3) / 16
c0 = (-162*alpha1 - 30*beta1 + 82*gamma1 + 81*alpha2 + 14*beta2 - 44*gamma2 - 54*alpha3 - 8*beta3 + 36*gamma3) / 16

# Define particular solution
psi = Matrix([(2*(-alpha1 + 2*gamma1)*x**2 + 2*(13*alpha1 + 2*beta1 - 8*gamma1 - alpha2 + 2*gamma2)*x + (-34*alpha1 - 6*beta1 + 18*gamma1 + 13*alpha2 + 2*beta2 - 8*gamma2 - 2*alpha3 + 4*gamma3)) / 8,
              (2*(-27*alpha1 - 4*beta1 + 10*gamma1)*x**2 + 2*(27*alpha1 + 6*beta1 - 8*gamma1 - 27*alpha2 - 4*beta2 + 10*gamma2)*x + (-2*beta1 - 6*gamma1 + 27*alpha2 + 6*beta2 - 8*gamma2 - 54*alpha3 - 8*beta3 + 20*gamma3)) / 16,
              (2*(-27*alpha1 - 4*beta1 + 18*gamma1)*x**2 + 2*(81*alpha1 + 14*beta1 - 44*gamma1 - 27*alpha2 - 4*beta2 + 18*gamma2)*x + (-162*alpha1 - 30*beta1 + 82*gamma1 + 81*alpha2 + 14*beta2 - 44*gamma2 - 54*alpha3 - 8*beta3 + 36*gamma3)) / 16])

# Derivative
psi_prime = diff(psi, x)

# Non-homogeneous term
q = Matrix([alpha1*x**2 + alpha2*x + alpha3,
            beta1*x**2 + beta2*x + beta3,
            gamma1*x**2 + gamma2*x + gamma3])

# Matrix A
A = Matrix([[4, 2, -2], [-27, -9, 11], [0, 1, -1]])

# Compute A * psi + q
lhs = A * psi + q

# Check difference
difference = simplify(lhs - psi_prime)
print("Difference (should be zero if correct):")
print(difference)
if difference == Matrix([0, 0, 0]):
    print("Verification successful: The particular solution satisfies the system.")
else:
    print("Verification failed: Check the coefficients or matrix A.")

# Output the particular solution in LaTeX
print("\nParticular solution in LaTeX:")
print("\\[")
print("\\left\\{")
print(f"\\begin{{aligned}}")
print(f"\\psi_1^{{(1)}} &= {latex(psi[0])}, \\\\")
print(f"\\psi_2^{{(1)}} &= {latex(psi[1])}, \\\\")
print(f"\\psi_3^{{(1)}} &= {latex(psi[2])}.")
print(f"\\end{{aligned}}")
print("\\right.")
print("\\]")

# Numeric test with example values
subs = {alpha1: 1, alpha2: 1, alpha3: 1, beta1: 2, beta2: 2, beta3: 2, gamma1: 3, gamma2: 3, gamma3: 3}
psi_num = psi.subs(subs)
print("\nNumeric solution with example values:")
print(psi_num)