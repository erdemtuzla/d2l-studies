import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# ==========================================
# 1- THE SOFTMAX FUNCTION
# ==========================================
# The goal of Softmax is to turn arbitrary numbers (logits) into valid probabilities.
# 1. They must be positive (solved by exp).
# 2. They must sum to 1.0 (solved by dividing by sum).

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# sum(0): Sums down the rows (collapsing the vertical axis). Result shape: [3]
# sum(1): Sums across columns (collapsing the horizontal axis). Result shape: [2]
# keepdims=True: Keeps the dimension as 1 instead of removing it (Shape: [2, 1]).
#                This is CRITICAL for broadcasting division later.
print("Sum across columns (keepdims=True):", X.sum(1, keepdims=True))

def softmax(X):
    # Step 1: Exponentiate. e^x makes everything positive.
    # e^0 = 1, e^negative = small positive number.
    X_exp = torch.exp(X)
    
    # Step 2: Calculate the partition function (normalization constant).
    # We sum across axis 1 (columns) to get the total "energy" for each row (image).
    partition = X_exp.sum(1, keepdims=True)
    
    # Step 3: Normalize.
    # Broadcasting happens here: PyTorch divides every element in the row by that row's sum.
    return X_exp / partition

# Test the function
X = torch.rand((2, 5)) # Create 2 random samples with 5 classes each.
X_prob = softmax(X)

# Verify: The probabilities for each sample should sum to exactly 1.0.
print("\nSoftmax Probabilities:\n", X_prob)
print("Sum of rows (should be 1.0):", X_prob.sum(1))

# ==========================================
# 2- THE MODEL
# ==========================================

class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize Weights (W)
        # Shape: [784, 10] -> [Pixels, Classes]
        # We start with random numbers (normal distribution) to break symmetry.
        # requires_grad=True: Tells PyTorch "Track every math operation on this variable 
        #                     so we can calculate gradients (backpropagation) later."
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        
        # Initialize Bias (b)
        # Shape: [10] -> One bias value per class.
        # Typically initialized to zeros.
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        # The optimizer needs a list of all variables it is allowed to change.
        return [self.W, self.b]
    
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    # Flatten the image:
    # Input X is [Batch_Size, 1, 28, 28].
    # We reshape to [Batch_Size, 784]. "-1" tells PyTorch to figure out the batch size automatically.
    X = X.reshape((-1, self.W.shape[0]))
    
    # The Linear Transformation: Y = XW + b
    # This matrix multiplication produces the "logits" (raw scores).
    logits = torch.matmul(X, self.W) + self.b
    
    # Apply Softmax to turn scores into probabilities.
    return softmax(logits)

# ==========================================
# 3- THE CROSS ENTROPY LOSS
# ==========================================
# We need to extract the predicted probability ONLY for the correct class label.

y = torch.tensor([0, 2]) # True labels for 2 examples: Class 0 and Class 2.
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # Model predictions.

# Fancy Indexing Explanation:
# range(2) creates [0, 1].
# y is [0, 2].
# y_hat[[0, 1], [0, 2]] selects:
#   Row 0, Col 0 (Value 0.1)
#   Row 1, Col 2 (Value 0.5)
# This efficiently grabs "The probability the model assigned to the right answer."
print("\nProbabilities assigned to correct classes:", y_hat[[0, 1], y])


def cross_entropy(y_hat, y):
    # 1. Select the probability of the correct label.
    # 2. Take the Logarithm.
    # 3. Negate it (because we want to MINIMIZE loss, but maximize likelihood).
    # 4. Take the mean over the batch.
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

print("Calculated Loss:", cross_entropy(y_hat, y))

@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)

# ==========================================
# 4- TRAINING
# ==========================================

data = d2l.FashionMNIST(batch_size=256)

# num_inputs=784 because 28 * 28 pixels = 784 features.
# num_outputs=10 because there are 10 classes (T-shirt, Dress, etc.).
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)

# Train for 10 epochs (passes through the dataset).
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)

# Force the training plot to stay open.
plt.show()

# ==========================================
# 5- PREDICTION & VISUALIZATION
# ==========================================
print("\nGenerating Predictions for visualization...")

# Get one batch of validation data
X, y = next(iter(data.val_dataloader()))

# 1. Get Predictions
# model(X) returns probabilities [Batch, 10].
# argmax(axis=1) picks the index with the highest probability.
preds = model(X).argmax(axis=1)

# 2. Find Mistakes (Masking)
# We cast preds to the same type as y to perform element-wise comparison.
# 'wrong' is a boolean tensor: True if prediction was wrong, False if correct.
wrong = preds.type(y.dtype) != y

# 3. Filter the batch
# We slice X, y, and preds using the boolean mask.
# Now we only have the data for the images the model got WRONG.
X, y, preds = X[wrong], y[wrong], preds[wrong]

# 4. Create Labels
# zip() iterates through the true labels and predicted labels simultaneously.
# We format the string to show "True" on the first line and "Pred" on the second line.
labels = [f'True: {a}\nPred: {b}' for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]

# 5. Visualize
# We pass the filtered images and our custom formatted labels.
# Limiting to first 8 errors to fit on screen nicely.
data.visualize([X, y], labels=labels)

print(f"Showing {len(labels)} misclassified images.")
plt.show()