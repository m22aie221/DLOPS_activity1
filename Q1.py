# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:11:48 2024

@author: ranji
"""

import numpy as np
import matplotlib.pyplot as plt


# Generate sample points
x = np.linspace(-5, 5, 500)

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

plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.title('Sigmoid')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, y_relu, label='ReLU', color='red')
plt.title('ReLU')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='green')
plt.title('Leaky ReLU')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, y_tanh, label='Tanh', color='purple')
plt.title('Tanh')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.tight_layout()
plt.show()
