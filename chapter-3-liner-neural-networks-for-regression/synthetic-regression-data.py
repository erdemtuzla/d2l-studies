import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# ==========================================
# 3.3.1 GENERATING THE DATASET
# ==========================================

class SyntheticRegressionData(d2l.DataModule):
    """
    Synthetic data for linear regression.
    
    INHERITANCE:
    Inherits from 'd2l.DataModule'. In Deep Learning, a 'DataModule' is a 
    standardized way to organize data loading. It handles downloading, 
    processing, and splitting data so the main training loop stays clean.
    """
    
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        """
        Initializes the fake dataset environment.
        
        VARIABLES EXPLAINED:
        --------------------
        w (Weights): The 'Slope' or importance of each feature. 
                     If w=[2, -3.4], it means:
                     - Feature 1 increases result by 2.
                     - Feature 2 decreases result by 3.4.
        
        b (Bias):    The 'Intercept'. Where the line starts on the Y-axis 
                     when all inputs are 0.
        
        noise:       Randomness added to the data. Real world data is never 
                     perfect. We add noise to make the problem realistic 
                     (stochastic).
        
        num_train:   Number of examples used for TRAINING (teaching the AI).
        
        num_val:     Number of examples used for VALIDATION (testing the AI).
                     We keep these separate to ensure the AI isn't just 
                     memorizing answers.
                     
        batch_size:  How many data points the AI studies at one time. 
                     We don't feed all 1000 rows at once (too heavy). 
                     We feed them in small "batches" (chunks) of 32.
        """
        super().__init__()
        
        # Saves all the arguments (w, b, noise, etc.) into 'self' 
        # automatically so we can use them later.
        self.save_hyperparameters()
        
        # Calculate total number of rows (samples) needed
        n = num_train + num_val
        
        # ---------------------------------------------------------
        # GENERATE FEATURES (X)
        # ---------------------------------------------------------
        # self.X: The Input Matrix (Features).
        # Shape: (n rows, len(w) columns).
        # torch.randn: Generates numbers from a Normal Distribution (Bell Curve).
        #              Mean=0, Variance=1.
        self.X = torch.randn(n, len(w))
        
        # ---------------------------------------------------------
        # GENERATE NOISE
        # ---------------------------------------------------------
        # Generate random errors to add to our perfect math formula.
        # Shape: (n rows, 1 column).
        noise = torch.randn(n, 1) * noise
        
        # ---------------------------------------------------------
        # GENERATE LABELS (y)
        # ---------------------------------------------------------
        # self.y: The Target Vector (Labels). This is the "Answer Key".
        # Formula: y = X * w + b + noise
        #
        # torch.matmul: Matrix Multiplication.
        #               Multiplies the Input Matrix (X) by the Weights (w).
        #
        # w.reshape((-1, 1)): Reshaping is crucial here. 
        #                     'w' usually starts as a flat list [2, -3.4].
        #                     We need it to be a column vector [[2], [-3.4]] 
        #                     to do matrix math correctly.
        #                     -1 tells PyTorch: "Figure out this dimension automatically".
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

# USAGE EXAMPLE:
# Create a dataset where 1st feature is multiplied by 2, 2nd by -3.4, plus 4.2 bias.
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)

# Print the first row to see what inputs (X) and answer (y) look like.
print("Features (Row 0): ", data.X[0]) 
print("Label (Answer 0): ", data.y[0])


# ==========================================
# 3.3.2 READING THE DATASET (MANUAL IMPLEMENTATION)
# ==========================================

@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    """
    A 'Generator' function that yields batches of data.
    
    DECORATOR (@d2l.add_to_class):
    This allows us to add this function to the 'SyntheticRegressionData' class 
    we defined above, without having to rewrite the whole class. 
    It's a way to organize code in educational notebooks.
    
    ARGS:
    train (bool): If True, we return training data. If False, validation data.
    """
    
    if train:
        # Create a list of row numbers [0, 1, 2, ... 999]
        indices = list(range(0, self.num_train))
        
        # SHUFFLING:
        # Crucial Step! We randomize the order of data.
        # If we don't shuffle, the AI might learn the *order* of answers 
        # instead of the underlying pattern (like memorizing "Answer C" 
        # usually follows "Answer B").
        random.shuffle(indices)
    else:
        # For validation (testing), order doesn't matter as much, 
        # so we just take the next chunk of indices [1000, 1001, ...].
        indices = list(range(self.num_train, self.num_train + self.num_val))
    
    # ---------------------------------------------------------
    # THE BATCH LOOP
    # ---------------------------------------------------------
    # Iterate through the list of indices, jumping by 'batch_size' (e.g., 32) steps.
    for i in range(0, len(indices), self.batch_size):
        
        # select a small group of row IDs (e.g., row 0 to 32)
        batch_indices = torch.tensor(indices[i: i + self.batch_size])
        
        # YIELD vs RETURN:
        # 'yield' makes this a Generator. It returns the batch, pauses execution,
        # and remembers where it left off.
        # This saves memory. We don't load the whole dataset into RAM; 
        # we serve it one spoon (batch) at a time.
        yield self.X[batch_indices], self.y[batch_indices]

# USAGE EXAMPLE:
# Get the first batch (spoonful) of data.
# 'next(iter(...))' triggers the generator to run once and stop.
X, y = next(iter(data.train_dataloader()))
print("\nManual Loader Batch Shapes:")
print("X shape (Inputs): ", X.shape) # Should be [32, 2] (32 rows, 2 features)
print("y shape (Labels): ", y.shape) # Should be [32, 1] (32 rows, 1 answer)


# ==========================================
# 3.3.3 CONCISE IMPLEMENTATION (PROFESSIONAL WAY)
# ==========================================

@d2l.add_to_class(d2l.DataModule)
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    """
    A helper method to use PyTorch's built-in tools.
    Instead of writing 'yield' loops manually, we use 'DataLoader'.
    """
    # Filter the tensors (X and y) to only include the rows we want (train or val)
    tensors = tuple(a[indices] for a in tensors)
    
    # TensorDataset:
    # A PyTorch wrapper that glues X and y together. 
    # It ensures that if we grab row 5 of X, we also get row 5 of y.
    dataset = torch.utils.data.TensorDataset(*tensors)
    
    # DataLoader:
    # The industry-standard tool. It handles:
    # 1. Shuffling (if shuffle=True)
    # 2. Batching (slicing data into chunks)
    # 3. Multiprocessing (loading data in parallel for speed)
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)

@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    """
    Overwrite the previous manual 'get_dataloader' with the professional version.
    """
    # SLICE:
    # A python object representing a range.
    # If train is True: slice(0, 1000) -> Rows 0 to 1000.
    # If train is False: slice(1000, None) -> Rows 1000 to the end.
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    
    # Call the helper defined above
    return self.get_tensorloader((self.X, self.y), train, i)

# USAGE EXAMPLE:
X, y = next(iter(data.train_dataloader()))
print('\nConcise Loader Batch Shapes:')
print('X shape:', X.shape, '\ny shape:', y.shape)

# Check how many batches exist. 
# If we have 1000 items and batch_size is 32, we expect 31 or 32 batches.
print("Total number of batches available:", len(data.train_dataloader()))