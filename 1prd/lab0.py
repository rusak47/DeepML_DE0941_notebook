import numpy as np
import matplotlib.pyplot as plt

# f(x) = sin(2x) + 2e^(3x)
def f(x):
    return np.sin(2*x) + 2*np.power(np.e, 3*x)


x = np.arange(-2, 2.1, 0.1)
y = f(x)
plt.figure(figsize=(10, 4))
plt.title('f(x) = sin(2x) + 2e^(3x)')
plt.xlabel('x')
plt.ylim(-50, 850)
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()