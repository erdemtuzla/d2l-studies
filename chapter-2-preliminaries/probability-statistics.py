import torch
import matplotlib.pyplot as plt
import random

from torch.distributions.multinomial import Multinomial
# from d2l import torch as d2l

print("### PROBABILITY AND STATISTICS ###\n")

num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print(f"Heads: {heads}, Tails: {tails}")

fair_probs = torch.tensor([0.5, 0.5])
print(Multinomial(10000, fair_probs).sample())

# Simulate 10,000 flips
counts = Multinomial(1, fair_probs).sample((10000,))

# Calculate cumulative counts
cumulative_counts = counts.cumsum(dim=0)
# print(cumulative_counts)

# Normalize to get probabilities
estimates = cumulative_counts / cumulative_counts.sum(dim=1, keepdim=True)
estimates = estimates.numpy()

# Plot the results
plt.figure(figsize=(4.5, 3.5))
plt.plot(estimates[:, 0], label="P(Head)")
plt.plot(estimates[:, 1], label="P(Tail)")

# Add the dashed line to 0.5
plt.axhline(y=0.5, color='black', linestyle='dashed')
# Labels and Legend
plt.xlabel('Samples')
plt.ylabel('Estimated probability')
plt.legend()
plt.grid(True)

# Show the plot window
# plt.show()

