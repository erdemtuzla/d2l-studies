import torch
from d2l import torch as d2l

# ==========================================
# 1- THE CLASSIFIER CLASS
# ==========================================

# We define a base class for classification models.
# It inherits from d2l.Module (which wraps torch.nn.Module) to handle 
# the training loop boilerplate (plotting, logging, etc.).
class Classifier(d2l.Module):
    
    # This method is called automatically during the validation phase of training.
    # Its job is to calculate and log how well the model is doing on unseen data.
    def validation_step(self, batch):
        # Unpacking the batch:
        # batch[:-1] -> The inputs (X). We use * to unpack them if there are multiple inputs.
        # batch[-1]  -> The labels (Y). The last item is always the target.
        # self(...)  -> Calls the forward() method to get predictions.
        Y_hat = self(*batch[:-1])
        
        # Calculate and plot the Loss (Error).
        # train=False tells the plotter that this is a validation curve (usually a dashed line).
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        
        # Calculate and plot the Accuracy.
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
        
# We add the optimizer configuration to the base Module class.
# This tells the Trainer how to update the model's weights.
@d2l.add_to_class(d2l.Module)
def configure_optimizers(self):
    # SGD (Stochastic Gradient Descent): The standard algorithm for updating weights.
    # self.parameters(): Returns all learnable weights (w) and biases (b) in the model.
    # lr=self.lr: The Learning Rate. Controls how big of a step we take during an update.
    return torch.optim.SGD(self.parameters(), lr=self.lr)

# ==========================================
# 2- ACCURACY
# ==========================================

# This method calculates how many predictions matched the actual labels.
@d2l.add_to_class(Classifier)  # @save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    
    # 1. Reshape Y_hat (Predictions)
    # Y_hat shape comes in as [Batch_Size, Num_Classes] (e.g., [64, 10]).
    # .reshape((-1, ...)) ensures it is a 2D matrix even if the batch dimensions are weird.
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    
    # 2. Find the Winner (Argmax)
    # Y_hat contains probabilities/logits (e.g., [0.1, 0.8, 0.1]).
    # .argmax(axis=1): Finds the INDEX of the highest value (e.g., index 1).
    # .type(Y.dtype): Casts the result to the same data type as the labels (usually int64/long).
    preds = Y_hat.argmax(axis=1).type(Y.dtype)
    
    # 3. Compare Predictions vs Reality
    # Y.reshape(-1): Flattens the labels into a 1D vector to match 'preds'.
    # (preds == Y...): Creates a boolean tensor [True, False, True...].
    # .type(torch.float32): Converts booleans to floats (True -> 1.0, False -> 0.0).
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    
    # 4. Return the Result
    # If averaged=True: Returns the mean accuracy (e.g., 0.85 for 85%).
    # If averaged=False: Returns the list of 1s and 0s (used if we want to count total correct later).
    return compare.mean() if averaged else compare