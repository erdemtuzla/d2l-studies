import torch

x = torch.arange(4.0)
print(x)

x.requires_grad_(True)

# Gradient is none by default
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

# Take the gradient of Y with respect to X by calling its backward method
y.backward()

print(x.grad)

print(x.grad == 4 * x)

# PyTorch doesn't reset the gradient buffer automatically.
# Instead, the new gradient added to the already-stored one
# This becomes handy when we want to optimize the sum of multiple objective functions

x.grad.zero_()
y = x.sum()
print(y)
y.backward()
print(x.grad)

print("\n# BACKWARD FOR NON-SCALAR VARIABLES #\n")

x.grad.zero_()
y = x * x
print(y)

y.backward(gradient=torch.ones(len(y)))
print(x.grad)

print("\n# DETACHING COMPUTATIONS #\n")
x.grad.zero_()
y = x * x
print(f"y: {y}")
u = y.detach()
z = u * x

print(f"u: {u}")
print(f"z: {z}")

z.sum().backward()

print(x.grad == u)

print("\n# GRADIENTS AND PYTHON CONTROL FLOW #\n")

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)

d.backward()

print(f"a: {a}")
print(f"d: {d}")

print(f"a.grad: {a.grad}") 
print(f"d/a: {d/a}")

# END