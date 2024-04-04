# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:11:48 2024

@author: ranji
"""

import numpy as np
import matplotlib.pyplot as plt


# Generate sample points
x_random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])
x = x_random_values

# Get out for sigmoid funtion
y_sigmoid = 1 / (1 + np.exp(-x))


# get output for Relu
y_relu = np.maximum(0, x)

# get output for Leaky Relu
a = 0.09 
y_leaky_relu = np.where(x > 0, x, a * x)

# get output ofr tanh
y_tanh = np.tanh(x)

# Plot graphs
plt.figure(figsize=(15, 10))

plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.title('Sigmoid')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()


plt.tight_layout()
plt.show()
