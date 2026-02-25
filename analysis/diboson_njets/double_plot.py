import numpy as np
import matplotlib.pyplot as plt


# --- Data ---
data1 = { #Run3
    "[2]": 0.9955083,
    "[3]": 1.25801749,
    "[4]": 1.98090739,
    "[5]": 2.14292729,
}
data1_fit = {
    "[2]": 0.98569395,
    "[3]": 1.34159885,
    "[4]": 1.69750375,
    "[5]": 2.05340866,
    "[6]": 2.409314,
    "[7]": 2.765218,
}
yerr1 = [0.01780174, 0.03716305, 0.1003777, 0.25098211] 
# data2 = {
#     "[2]": 1.1999534110679877,
#     "[3]": 1.4648336628132348,
#     "[4]": 1.7384550502594722,
#     "[5]": 2.5367659601551021,
# }
# data2_fit = {
#     "[2]": 1.1972716694861654,
#     "[3]": 1.4825555284619116,
#     "[4]": 1.767839387437658,
#     "[5]": 2.053123246413404,
#     "[6]": 2.33840710538915,
#     "[7]": 2.6236909643648962,
# }
data3 = {
    "[2]": 0.95323171,
    "[3]": 1.14964901,
    "[4]": 1.40249826,
    "[5]": 1.77996707,
}
data3_fit = {
    "[2]": 0.95002636,
    "[3]": 1.16969909,
    "[4]": 1.38937181,
    "[5]": 1.60904454,
    "[6]": 1.828717,
    "[7]": 2.04839,
}
yerr3 = [0.01201504, 0.02371558, 0.05292038, 0.13969771]
data4 = {
    "[2]": 0.96048623,
    "[3]": 1.17096204,
    "[4]": 1.44722493,
    "[5]": 1.85825373,
}
data4_fit = {
    "[2]": 0.86572173,
    "[3]": 1.16853894,
    "[4]": 1.47135616,
    "[5]": 1.77417338,
    "[6]": 2.07699059,
    "[7]": 2.37980781,
}
# --- Convert keys to numeric x-values ---
# We'll take the midpoint of each interval label, e.g. "[0,1]" → 0.5
def keys_to_numbers(keys):
    return [float(k.strip("[]")) for k in keys]
x2to7 = keys_to_numbers(data1_fit.keys())
x1 = keys_to_numbers(data1.keys())
y1 = list(data1.values())

x2 = keys_to_numbers(data2.keys())
y2 = list(data2.values())

x3 = keys_to_numbers(data3.keys())
y3 = list(data3.values())

x4 = keys_to_numbers(data4.keys())
y4 = list(data4.values())
# --- Fit lines (linear regression) ---
coeffs1 = np.polyfit(x1, y1, 1)
coeffs2 = np.polyfit(x2, y2, 1)
coeffs3 = np.polyfit(x3, y3, 1)
fit1 = np.poly1d(coeffs1)
fit2 = np.poly1d(coeffs2)
fit3 = np. poly1d(coeffs3)
# --- Plot ---
plt.figure(figsize=(8, 6))
plt.errorbar(
    x1,
    y1,
    yerr=yerr1,
    fmt='o',        # point marker only
    markersize=6,
    capsize=4,      # little caps at the end of bars
    linestyle='',   # no connecting line (scatter-like)
    color='blue'
)
plt.scatter(x1, y1, color='blue', label='Run3')
plt.plot(x2to7, list(data1_fit.values()), color='blue', label=f'Run3_fit')
#plt.scatter(x2, y2, color='red', label='Run2_official')
#plt.plot(x2to7, list(data2_fit.values()), color='red', label=f'Run2_official_fit')
plt.scatter(x3, y3, color = 'green', label = 'Run2_fit')
plt.plot(x2to7, list(data3_fit.values()), color = 'green', label = 'Run2_fit')
plt.errorbar(
    x3,
    y3,
    yerr=yerr3,
    fmt='o',        # point marker only
    markersize=6,
    capsize=4,      # little caps at the end of bars
    linestyle='',   # no connecting line (scatter-like)
    color='green'
)
#plt.scatter(x3, y3, color='green',  label='Run2_recreated')
#plt.plot(x3, list(data3_fit.values()), color='green', label=f'Run2_recreated_fit')

plt.title("Diboson SF")
plt.xlabel("njets")
plt.ylabel("SF")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("two_fits.png", dpi=300)  # Save as PNG at high resolution


plt.show()

