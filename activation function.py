import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
p1 = plt.subplot(311)
y = tanh(x)
p1.plot(x, y)
p1.set_title('tanh')
p1.axhline(ls='--', color='r')
p1.axvline(ls='--', color='r')


p2 = plt.subplot(312)
y = sigmoid(x)
p2.plot(x, y)
p2.set_title('sigmoid')
p2.axhline(0.5, ls='--', color='r')
p2.axvline(ls='--', color='r')

p3 = plt.subplot(313)
y = relu(x)
p3.plot(x, y)
p3.set_title('relu')
p3.axvline(ls='--', color='r')

plt.subplots_adjust(hspace=1)
plt.show()
