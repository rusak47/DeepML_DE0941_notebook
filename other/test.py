import numpy as np
import matplotlib.pyplot as plt

def angle_to_exp_coef(angle_deg, max_angle_deg=90, k=5):
    #x = np.clip(angle_deg / max_angle_deg, 0, 1)  # Normalize and clamp
    #x = np.clip(angle_deg / max_angle_deg, -1, 1)  # Normalize and clamp
    #x = np.abs(angle_deg) / max_angle_deg  # Normalize and clamp
    x = angle_deg / max_angle_deg  # Normalize and clamp

    return ((np.exp(k * x) - 1) / (np.exp(k) - 1))*10
    #return (np.exp(k * np.abs(x)) - 1) / (np.exp(k) - 1)
    #return (np.exp(k * x) ) / (np.exp(k) )


#angles = np.linspace(0, 90, 500)
max_angle = 180
angles = np.linspace(0, max_angle, 500)
coefs1 = angle_to_exp_coef(angles, max_angle_deg=max_angle, k=1)
coefs = angle_to_exp_coef(angles, max_angle_deg=max_angle, k=7)
coefs14 = angle_to_exp_coef(angles, max_angle_deg=max_angle, k=14)

plt.plot(angles, coefs1)
plt.plot(angles, coefs)
plt.plot(angles, coefs14)
plt.title("Exponential Coefficient vs Angle")
plt.xlabel("Angle (degrees)")
plt.ylabel("Exponential Coefficient")
plt.grid(True)
plt.show()
