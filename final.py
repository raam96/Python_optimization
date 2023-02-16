import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

x = np.arange(1, 673)
# Read the input data from the CSV file
data = np.loadtxt('input.csv', delimiter=',').T
P_wt = data[0]
P_wec = data[1]
P_pv = data[2]
P_load = data[3]


# Define the constants and parameters
delta_t = 1
C_wt = 1.28 * 300000
C_dg = 1.28 * 25000
C_wec = 1.28 * 1000000
C_pv = 1.28 * 1250
C_bat = 1.28 * 546
C_fuel = 1.28
r = 0.06
om = 0.02
y_wt = 20
y_wec = 20
y_pv = 20
y_dg = 20
eta_inv = 0.97
eta_pv = 0.16
P_dg_max = 175
P_dg_min = 0
P_ren_max = 1750
P_ren_min = 0
y_bat = 5
eta_d_bat = 0.95
eta_c_bat = 0.95
C_bat_Max = 2000

# Define the optimization bounds
m = len(P_load)
print(m)
P_bat_ch_min = -300
P_bat_ch_max = 0
P_bat_disch_min = 0
P_bat_disch_max = 300
A_pv_min = 0
A_pv_max = 10000
C_bat_min = 0
C_bat_max = 2000
lb = [P_bat_ch_min] * m + [P_bat_disch_min] * m + [A_pv_min, C_bat_min]
ub = [P_bat_ch_max] * m + [P_bat_disch_max] * m + [A_pv_max, C_bat_max]

# Define the initial guess
x0 = np.zeros(2*m+2)

# Define the objective function
# Define the objective function
# Define the objective function
def objective(u):
    # Define the variables from the input vector
    P_bat_ch = u[:m]
    P_bat_disch = u[m:2*m]
    A_pv = u[2*m]
    C_bat_max = u[2*m+1]

    # Define the other variables you need to calculate the objective
    P_ren = P_wt + P_wec + A_pv * P_pv
    P_dg = np.zeros(m)
    P_gs = np.vstack([P_ren, P_bat_disch, P_dg])
    E_bat = np.ones(m)
    SOC_bat = np.full(m, 0.5 * (SOC_bat_min + SOC_bat_max))
    P_gen = np.zeros(m)
    dif = np.zeros(m)
    P_consumption = np.zeros(m)
    Bx = np.random.randint(0, 2, size=m)

    # Calculate the objective
    # Calculate the objective
    CRF_wt = r * (1 + r) ** y_wt / ((1 + r) ** y_wt - 1)
    NPC_wt = C_wt * (CRF_wt + om)
    CRF_wec = r * (1 + r) ** y_wec / ((1 + r) ** y_wec - 1)
    NPC_wec = C_wec * (CRF_wec + om)
    CRF_pv = r * (1 + r) ** y_pv / ((1 + r) ** y_pv - 1)
    NPC_pv = A_pv * C_pv * (P_pv * CRF_pv + om)
    NPC_pv = max(NPC_pv)
    CRF_bat = r * (1 + r) ** y_bat / ((1 + r) ** y_bat - 1)
    NPC_bat = C_bat_max * C_bat * (CRF_bat + om)
    A = 0.246
    B1 = 0.08145
    fuel_con = A * np.sum(P_dg) + B1 * P_dg_max
    fuel_dg = fuel_con * len(P_dg[P_dg != 0]) * C_fuel
    CRF_dg = r * (1 + r) ** y_dg / ((1 + r) ** y_dg - 1)
    NPC_dg = C_dg * (CRF_dg + om) + fuel_dg
    Total_NPC = NPC_wt + NPC_wec + NPC_pv + NPC_bat + NPC_dg
    LCOE = Total_NPC / np.sum(P_load * delta_t)
    f1 = LCOE
    P_supply = (P_ren + P_bat_ch + P_dg) * eta_inv
    DL = np.zeros(m)
    DL[P_supply > P_load] = (P_supply - P_load)[P_supply > P_load] * delta_t
    DR = np.sum(DL) / np.sum(P_ren[:len(DL)] * delta_t)
    f2 = DR
    g1 = E_bat - C_bat_max
    g2 = C_bat_min - E_bat
    g3 = SOC_bat - SOC_bat_max
    g4 = SOC_bat_min - SOC_bat
    g5 = P_bat_ch - P_bat_ch_max
    g6 = P_bat_ch_min - P_bat_ch
    g7 = P_bat_disch - P_bat_disch_max
    g8 = P_bat_disch_min - P_bat_disch
    g1[g1 < 0] = 0
    g2[g2 < 0] = 0
    g3[g3 < 0] = 0
    g4[g4 < 0] = 0
    g5[g5 < 0] = 0
    g6[g6 < 0] = 0
    g7[g7 < 0] = 0
    g8[g8 < 0] = 0
    h1 = np.abs(P_load[:len(P_ren)] - (P_ren + P_bat_disch + P_dg) * eta_inv)
    V = np.sum(g1) + np.sum(g2) + np.sum(g3) + np.sum(g4) + np.sum(g5) + np.sum(g6) + np.sum(g7) + np.sum(g8) + np.sum(
        h1)

    # Combine the objective and constraint functions
    c = [g1, g2, g3, g4, g5, g6, g7, g8, h1]
    penalty = 0
    for i in range(len(c)):
        for j in range(len(c[i])):
            if c[i][j] != 0:
                penalty += 1
                break
    f1 = LCOE + penalty * 1e8 + DR
    #f2 = DR
    return f1


P_bat_ch_best = np.zeros(m)
P_bat_disch_best = np.zeros(m)
A_pv_best = 0.0
C_bat_max_best = 0.0
SOC_bat_min = 0.5
SOC_bat_max = 0.8

# Set up the bounds
lb = np.concatenate((P_bat_ch_min * np.ones(m), P_bat_disch_min * np.ones(m), np.array([A_pv_min, C_bat_min])))
ub = np.concatenate((P_bat_ch_max * np.ones(m), P_bat_disch_max * np.ones(m), np.array([A_pv_max, C_bat_Max])))
bounds = list(zip(lb, ub))

# Set up the initial guess
x0 = np.concatenate((P_bat_ch_best * np.ones(m), P_bat_disch_best * np.ones(m), np.array([A_pv_best, C_bat_max_best])))

# Perform optimization
results = differential_evolution(objective, bounds, maxiter=10, popsize=30, disp=True, init='latinhypercube', seed=1)
u = results.x

# Calculate the best values
Bx_best = u[:m]
P_bat_ch_best = u[m:2 * m]
P_bat_disch_best = u[m:2 * m] * np.ones(m)
A_pv_best = u[-2]
C_bat_max_best = u[-1]
P_ren_best = P_wt + P_wec + A_pv_best * P_pv * np.ones(m)
P_dg_best = np.zeros(m)
P_gen_best = np.zeros(m)
dif = np.zeros(m)
P_consumption_best = np.zeros(m)
E_bat_best = np.ones(m)
SOC_bat_best = (SOC_bat_max / 2) * np.ones(m)

P_dg_best_resized = np.resize(P_dg_best, m)
P_gs_best = np.vstack((P_ren_best, P_bat_disch_best, P_dg_best_resized))
for k in range(m):
    P_supply = P_ren_best[k] + P_bat_ch_best[k] + P_dg_best_resized[k]
    if P_supply > P_load[k]:
        P_dg_best_resized[k] = P_supply - P_load[k]
        P_gs_best[2][k] = P_dg_best_resized[k]
    elif P_supply < P_load[k]:
        P_bat_disch_best[k] = P_load[k] - P_supply
        P_gs_best[1][k] = P_bat_disch_best[k]
    P_gen_best[k] = P_ren_best[k] + P_dg_best_resized[k]
    dif[k] = P_load[k] - P_gen_best[k]
    if dif[k] < 0:
        P_bat_ch_best[k] = -dif[k]
        P_gs_best[0][k] = P_ren_best[k] + P_bat_disch_best[k]
    P_consumption_best[k] = P_load[k] - P_bat_ch_best[k] - P_gs_best[0][k]

# Print the best values
print("Bx_best:", Bx_best)
print("P_dg_best:", P_dg_best)
print("P_bat_ch_best:", P_bat_ch_best)
print("P_bat_disch_best:", P_bat_disch_best)
print("P_ren_best:", P_ren_best)
print("P_gen_best:", P_gen_best)
print("P_gs_best:", P_gs_best)
print("P_consumption_best:", P_consumption_best)
print("E_bat_best:", E_bat_best)
print("SOC_bat_best:", SOC_bat_best)
print("A_pv_best:", A_pv_best)
print("C_bat_max_best:", C_bat_max_best)



# Plot 2
plt.plot(x, P_bat_ch_best, label="Charge")
plt.plot(x, P_bat_disch_best, label="Discharge")
plt.title("Best Battery Charge and Discharge Powers")
plt.xlabel("Time (hour)")
plt.ylabel("Power (kW)")
plt.legend()

plt.plot(x, P_ren_best, label="Renewable")
plt.plot(x, P_dg_best, label="Diesel")
plt.title("Best Renewable Energy and Diesel Generation")
plt.xlabel("Time (hour)")
plt.ylabel("Power (kW)")
plt.legend()

plt.plot(x, SOC_bat_best)
plt.title("Best Battery State of Charge")
plt.xlabel("Time (hour)")
plt.ylabel("State of Charge")
plt.plot(x, P_gen_best)
plt.title("Best Power Generation")
plt.xlabel("Time (hour)")
plt.ylabel("Power Generation (kW)")
# plt.plot(x, P_gs_best_resized[0], label="Renewable + Storage")
# plt.plot(x, P_gs_best_resized[1], label="Battery Discharge")
# plt.plot(x, P_gs_best_resized[2], label="Diesel")

plt.title("Best Power Supply from Microgrid")
plt.xlabel("Time (hour)")
plt.ylabel("Power (kW)")
plt.legend()

plt.show()