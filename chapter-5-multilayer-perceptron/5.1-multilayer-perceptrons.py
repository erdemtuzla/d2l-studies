import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 1. Generate the Data (Same as before)
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

# print(f"x: {x}")
# print(f"y: {y}")

# 2. Prepare Data for Matplotlib
# .detach(): Removes the tensor from the gradient calculation graph.
# .numpy(): Converts the PyTorch tensor to a standard NumPy array.
x_np = x.detach().numpy()
y_np = y.detach().numpy()

# 3. Create the Plot
# figsize=(5, 2.5) matches the d2l dimensions
plt.figure(figsize=(5, 2.5))

# Plot x vs y
plt.plot(x_np, y_np)

# Add labels and styling
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.title('ReLU Function')
plt.grid(True) # Adds the grid lines common in d2l plots

# Show the plot
plt.show()

##############################################
##############################################
##############################################

# 1. Compute the Gradients (Same as original)
# We pass torch.ones_like(x) because x is a vector, not a scalar.
# retain_graph=True is needed if you plan to call backward() again later.
y.backward(torch.ones_like(x), retain_graph=True)

# 2. Prepare Data for Matplotlib
x_np = x.detach().numpy()
grad_np = x.grad.numpy() # Convert the calculated gradients to numpy

# 3. Create the Plot
plt.figure(figsize=(5, 2.5))

# Plot x vs Gradients
plt.plot(x_np, grad_np)

# Add labels and styling
plt.xlabel('x')
plt.ylabel('grad of relu')
plt.title('Gradient of ReLU')
plt.grid(True)

# Show the plot
plt.show()

##############################################
##############################################
##############################################

# 1. Generate Data (Assuming x is defined as before)
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)

# 2. Prepare Data for Matplotlib
# .detach() removes the tensor from the gradient graph
# .numpy() converts it to a format matplotlib understands
x_np = x.detach().numpy()
y_np = y.detach().numpy()

# 3. Create the Plot
plt.figure(figsize=(5, 2.5))

# Plot x vs y
plt.plot(x_np, y_np)

# Add labels and styling
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
plt.grid(True) # Adds the grid lines

# Show the plot
plt.show()


##############################################
##############################################
##############################################


# 1. Setup (Recreating context from previous steps)
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)

# Simulate a previous backward pass so there is something to clear
# (This line usually exists in the d2l flow before the snippet you pasted)
y.backward(torch.ones_like(x), retain_graph=True)

# ==========================================================
# TRANSLATION OF YOUR SNIPPET STARTS HERE
# ==========================================================

# 2. Clear out previous gradients
# PyTorch accumulates gradients by default. If we don't zero them,
# the new gradients would be added to the old ones (doubling them).
if x.grad is not None:
    x.grad.data.zero_()

# 3. Calculate Gradients
y.backward(torch.ones_like(x), retain_graph=True)

# 4. Prepare Data for Matplotlib
x_np = x.detach().numpy()
grad_np = x.grad.numpy()

# 5. Create the Plot
plt.figure(figsize=(5, 2.5))
plt.plot(x_np, grad_np)
plt.xlabel('x')
plt.ylabel('grad of sigmoid')
plt.title('Gradient of Sigmoid')
plt.grid(True)
plt.show()

##############################################
##############################################
##############################################

