from sympy import symbols, solve, Eq

# Define symbolic variables
a0, a1, a2, a3, a4 = symbols('a0 a1 a2 a3 a4')
b0, b1, b2, b3, b4 = symbols('b0 b1 b2 b3 b4')
c0, c1, c2, c3, c4 = symbols('c0 c1 c2 c3 c4')  # c4 is a free parameter
alpha4, beta4, gamma4 = symbols('alpha4 beta4 gamma4')
alpha5, beta5, gamma5 = symbols('alpha5 beta5 gamma5')

# --- System for x^4 ---
print("--- Solving for x^4 coefficients (a4, b4, c4) ---")
eq_x4_1 = Eq(-6*a4 - 2*b4 + 2*c4, 0)
eq_x4_2 = Eq(27*a4 + 7*b4 - 11*c4, 0)
eq_x4_3 = Eq(-b4 - c4, 0)

sol_x4_raw = solve((eq_x4_1, eq_x4_2, eq_x4_3), (a4, b4, c4), dict=True)[0]
sol_x4_a4 = sol_x4_raw[a4]
sol_x4_b4 = sol_x4_raw[b4]

print(f"a4 = {sol_x4_a4}")
print(f"b4 = {sol_x4_b4}")
print(f"c4 = {c4} (free parameter)\n")

# --- System for x^3 ---
print("--- Solving for x^3 coefficients (a3, b3, c3) ---")
eq_x3_1 = Eq(-6*a3 + 4*sol_x4_a4 - 2*b3 + 2*c3, 0)
eq_x3_2 = Eq(27*a3 + 7*b3 + 4*sol_x4_b4 - 11*c3, 0)
eq_x3_3 = Eq(-b3 - c3 + 4*c4, 0)

sol_x3_raw = solve((eq_x3_1, eq_x3_2, eq_x3_3), (a3, b3, c3), dict=True)[0]
sol_x3_a3 = sol_x3_raw[a3]
sol_x3_b3 = sol_x3_raw[b3]
sol_x3_c3 = sol_x3_raw[c3]

print(f"a3 = {sol_x3_a3}")
print(f"b3 = {sol_x3_b3}")
print(f"c3 = {sol_x3_c3}\n")

# --- System for x^2 ---
print("--- Solving for x^2 coefficients (a2, b2, c2) ---")
eq_x2_1 = Eq(-6*a2 + 3*sol_x3_a3 - 2*b2 + 2*c2, 0)
eq_x2_2 = Eq(27*a2 + 7*b2 + 3*sol_x3_b3 - 11*c2, 0)
eq_x2_3 = Eq(-b2 - c2 + 3*sol_x3_c3, 0)

sol_x2_raw = solve((eq_x2_1, eq_x2_2, eq_x2_3), (a2, b2, c2), dict=True)[0]
sol_x2_a2 = sol_x2_raw[a2]
sol_x2_b2 = sol_x2_raw[b2]
sol_x2_c2 = sol_x2_raw[c2]

print(f"a2 = {sol_x2_a2}")
print(f"b2 = {sol_x2_b2}")
print(f"c2 = {sol_x2_c2}\n")

# --- System for x^1 ---
print("--- Solving for x^1 coefficients (a1, b1, c1) ---")
eq_x1_1 = Eq(-6*a1 + 2*sol_x2_a2 - 2*b1 + 2*c1, alpha4)
eq_x1_2 = Eq(27*a1 + 7*b1 + 2*sol_x2_b2 - 11*c1, beta4)
eq_x1_3 = Eq(-b1 - c1 + 2*sol_x2_c2, gamma4)

sol_x1_raw = solve((eq_x1_1, eq_x1_2, eq_x1_3), (a1, b1, c1), dict=True)[0]
sol_x1_a1 = sol_x1_raw[a1]
sol_x1_b1 = sol_x1_raw[b1]
sol_x1_c1 = sol_x1_raw[c1]

print(f"a1 = {sol_x1_a1}")
print(f"b1 = {sol_x1_b1}")
print(f"c1 = {sol_x1_c1}\n")

# --- System for x^0 ---
print("--- Solving for x^0 coefficients (a0, b0, c0) ---")
eq_x0_1 = Eq(-6*a0 + sol_x1_a1 - 2*b0 + 2*c0, alpha5)
eq_x0_2 = Eq(27*a0 + 7*b0 + sol_x1_b1 - 11*c0, beta5)
eq_x0_3 = Eq(-b0 - c0 + sol_x1_c1, gamma5)

sol_x0_raw = solve((eq_x0_1, eq_x0_2, eq_x0_3), (a0, b0, c0), dict=True)[0]
sol_x0_a0 = sol_x0_raw[a0]
sol_x0_b0 = sol_x0_raw[b0]
sol_x0_c0 = sol_x0_raw[c0]

print(f"a0 = {sol_x0_a0}")
print(f"b0 = {sol_x0_b0}")
print(f"c0 = {sol_x0_c0}")