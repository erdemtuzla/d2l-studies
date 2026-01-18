import math
import time
import numpy
import torch
import matplotlib.pyplot as plt

# Vector size
n = 10000

a = torch.ones(n)
b = torch.ones(n)

c = torch.zeros(n)

t = time.time()

for i in range(n):
    c[i] = a[i] + b[i]
    
print(f"Normal Loop: \t\t{time.time() - t:.5f} sec")


t = time.time()

d = a + b

print(f"Reloaded '+' operator: \t{time.time() - t:.5f} sec")

print("## THE NORMAL DISTRIBUTION AND SQUARED LOSS ##\n")

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * numpy.exp(-0.5 * (x - mu)**2 / sigma**2)

# 2. Data Setup
x = numpy.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]

# 3. Plotting with Matplotlib
plt.figure(figsize=(4.5, 2.5))  # Corresponds to figsize=(4.5, 2.5)

for mu, sigma in params:
    y = normal(x, mu, sigma)
    plt.plot(x, y, label=f'mean {mu}, std {sigma}')

# 4. Styling (Replicating d2l arguments)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()      # Automatically uses the 'label' defined in the loop
plt.grid(True)    # d2l usually adds a grid by default

plt.show()

