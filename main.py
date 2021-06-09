import numpy as np
import math
import matplotlib.pyplot as plt

x = np.linspace(-math.pi, math.pi, 2000)
y_sin = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 0.000001

for t in range(2000):
    y_polynomial_derived = a + b * x + c * x ** 2 + d * x ** 3
    loss = np.square(y_polynomial_derived - y_sin).sum()

    grad_y_polynomial_derived = 2.0 * (y_polynomial_derived - y_sin)

    grad_a = grad_y_polynomial_derived.sum()
    grad_b = (grad_y_polynomial_derived * x).sum()
    grad_c = (grad_y_polynomial_derived * x ** 2).sum()
    grad_d = (grad_y_polynomial_derived * x ** 3).sum()

    descent_grad_a = learning_rate * grad_a
    descent_grad_b = learning_rate * grad_b
    descent_grad_c = learning_rate * grad_c
    descent_grad_d = learning_rate * grad_d

    a = a - descent_grad_a
    b = b - descent_grad_b
    c = c - descent_grad_c
    d = d - descent_grad_d

y_polynomial_derived = a + b * x + c * x ** 2 + d * x ** 3

plt.plot(x, y_sin)
plt.plot(x, y_polynomial_derived)
plt.show()
