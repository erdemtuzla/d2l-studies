import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt 

# ==========================================
# 1. THE MODEL (The "Brain")
# ==========================================

class LinearRegressionScratch(d2l.Module):
    """
    The Linear Regression Model.
    Inherits from d2l.Module (which wraps torch.nn.Module).
    """
    def __init__(self, num_inputs, lr, sigma=0.01):
        """
        Initialize the model's memory (parameters).
        
        ARGS:
        - num_inputs: How many features does the data have? (e.g., 2: size & rooms).
        - lr: Learning Rate. How big of a step to take when learning.
        - sigma: Standard Deviation. Controls how "spread out" the initial random weights are.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # PARAMETER 1: WEIGHTS (w)
        # We start with random guesses. 
        # torch.normal(0, sigma...): Random numbers from a bell curve centered at 0.
        # size=(num_inputs, 1): A column vector (e.g., 2 rows, 1 column).
        # requires_grad=True: CRITICAL! This tells PyTorch to watch this variable 
        #                     and track all math operations done on it. This allows 
        #                     us to calculate the gradient (slope) later.
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        
        # PARAMETER 2: BIAS (b)
        # The y-intercept. We usually initialize this to 0.
        self.b = torch.zeros(1, requires_grad=True)

@d2l.add_to_class(LinearRegressionScratch)
def forward(self, X):
    """
    The Prediction Step (Forward Pass).
    Calculates: y = Xw + b
    
    ARGS:
    - X: The input data matrix.
    """
    # torch.matmul: Matrix multiplication. Multiplies inputs by weights.
    # + self.b: Adds the bias. PyTorch automatically "broadcasts" (stretches) 
    #           this single number to match the shape of the batch.
    return torch.matmul(X, self.w) + self.b

@d2l.add_to_class(LinearRegressionScratch)
def loss(self, y_hat, y):
    """
    The Error Calculation.
    Measures how wrong the model is using Mean Squared Error (MSE).
    
    ARGS:
    - y_hat: The model's prediction (calculated in forward()).
    - y: The actual correct answer (from the dataset).
    """
    # 1. Calculate the difference (error) squared.
    #    We divide by 2 so that when we take the derivative later (calculus), 
    #    the '2' from the exponent cancels out nicely: d/dx(x^2/2) = x.
    l = (y_hat - y) ** 2 / 2
    
    # 2. Return the average error of the batch.
    return l.mean()

# ==========================================
# 2. THE OPTIMIZER (The "Teacher")
# ==========================================

class SGD(d2l.HyperParameters):
    """
    Stochastic Gradient Descent (SGD).
    This class is responsible for updating the weights to reduce error.
    """
    def __init__(self, params, lr):
        """
        ARGS:
        - params: A list of the model's parameters [w, b] that need updating.
        - lr: Learning Rate. If this is too big, the model overshoots. 
              If too small, it learns forever.
        """
        self.save_hyperparameters()

    def step(self):
        """
        The Update Step.
        Adjusts weights in the opposite direction of the gradient.
        """
        for param in self.params:
            # param.grad: The 'slope' of error calculated by PyTorch.
            #             Points in the direction that INCREASES error.
            # We subtract (-) because we want to DECREASE error.
            # Update Rule: New_Weight = Old_Weight - (Learning_Rate * Gradient)
            param -= self.lr * param.grad

    def zero_grad(self):
        """
        Cleans the slate.
        PyTorch accumulates gradients by adding them up. If we don't zero them 
        out after every step, the new gradient is added to the old one, 
        creating a mess.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

@d2l.add_to_class(LinearRegressionScratch)
def configure_optimizers(self):
    """Binds the optimizer to the model."""
    return SGD([self.w, self.b], self.lr)


# ==========================================
# 3. THE TRAINING LOOP (The "Classroom")
# ==========================================

@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self, batch):
    """Helper to unpack data."""
    return batch

@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    """
    Runs one full pass (epoch) over the entire training dataset.
    """
    # Initialize storage for plotting (Local Fix)
    if not hasattr(self, 'loss_history'):
        self.loss_history = []
        
    # Switch model to 'Training Mode' (enables gradients)
    self.model.train()
    
    # Loop through the data in chunks (batches)
    for batch in self.train_dataloader:
        
        # 1. FORWARD PASS & LOSS
        # Ask model for prediction and calculate how wrong it was.
        loss = self.model.training_step(self.prepare_batch(batch))
        
        # 2. CLEAN UP
        # Delete old gradients from the previous batch.
        self.optim.zero_grad()
        
        # 3. BACKWARD PASS (Backpropagation)
        # torch.no_grad(): A safety wrapper. We wrap the update steps in this 
        #                  because we don't want PyTorch to track the update math 
        #                  itself as part of the gradient graph.
        with torch.no_grad():
            # Calculate gradients (find out which weights caused the error)
            loss.backward()
            
            # (Optional) Clip gradients prevents crashes if error is massive.
            if self.gradient_clip_val > 0:
                self.clip_gradients(self.gradient_clip_val, self.model)
            
            # 4. UPDATE PARAMETERS
            # Nudge the weights to be slightly better.
            self.optim.step()
            
            # Store the loss value for our plot
            self.loss_history.append(loss.item())
            
        self.train_batch_idx += 1
    
    # Validation Loop (Testing on unseen data)
    if self.val_dataloader is None:
        return
    self.model.eval() # Switch to evaluation mode (no gradients needed)
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1

# ==========================================
# 4. EXECUTION (Running the Experiment)
# ==========================================

# 1. Create the model
# Inputs=2 because we will generate data with 2 features.
# lr=0.03 means we take medium-sized steps.
model = LinearRegressionScratch(2, lr=0.03)

# 2. Create Synthetic Data (The "Correct Answer")
# We create a fake world where: y = 2*x1 - 3.4*x2 + 4.2
# The model doesn't know these numbers (2, -3.4, 4.2). It has to guess them.
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)

# 3. Create the Trainer
# max_epochs=3 means we will go through the entire dataset 3 times.
trainer = d2l.Trainer(max_epochs=3)

# 4. Train!
print("Starting training...")
trainer.fit(model, data)

# 5. Verify Results
print("\n--- Final Results ---")
with torch.no_grad():
    # Check how close our learned 'w' is to the real 'w' ([2, -3.4])
    print(f'Error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    # Check how close our learned 'b' is to the real 'b' (4.2)
    print(f'Error in estimating b: {data.b - model.b}')

# 6. Plotting (The Visualization)
print("Plotting loss graph...")
plt.figure(figsize=(8, 5))
plt.plot(trainer.loss_history, label='Training Loss')
plt.xlabel('Iterations (Batches)')
plt.ylabel('Loss (Mean Squared Error)')
plt.title('How the AI Learned (Loss over Time)')
plt.legend()
plt.grid(True)
plt.show()