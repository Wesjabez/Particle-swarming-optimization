import numpy as np
import matplotlib.pyplot as plt


m = 0.06        # max camber
p = 0.47642227  # camber position
t = 0.08        # thickness

# Number of points
num_points = 200
x = np.linspace(0, 1, num_points)

# Thickness distribution (NACA formula)
yt = 5 * t * (
    0.2969 * np.sqrt(x)
    - 0.1260 * x
    - 0.3516 * x**2
    + 0.2843 * x**3
    - 0.1015 * x**4
)


yc = np.zeros_like(x)
dyc_dx = np.zeros_like(x)

for i in range(len(x)):
    if x[i] < p:
        yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
        dyc_dx[i] = (2 * m / p**2) * (p - x[i])
    else:
        yc[i] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x[i] - x[i]**2)
        dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])

theta = np.arctan(dyc_dx)

# Upper surface
xu = x - yt * np.sin(theta)
yu = yc + yt * np.cos(theta)

# Lower surface
xl = x + yt * np.sin(theta)
yl = yc - yt * np.cos(theta)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(xu, yu, label="Upper Surface")
plt.plot(xl, yl, label="Lower Surface")
plt.plot(x, yc, '--', label="Camber Line")

plt.title("Optimized Airfoil Shape (PSO Result)")
plt.xlabel("Chord (x)")
plt.ylabel("Thickness / Camber")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()