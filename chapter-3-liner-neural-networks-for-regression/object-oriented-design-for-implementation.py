import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


### 3.2.1 UTILITIES ###


def add_to_class(Class):
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class A:
    def __init__(self):
        self.b = 1

a = A()

@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()

class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
    
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)

# --- 1. A Local Replacement for ProgressBoard ---
class LocalProgressBoard:
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None):
        """
        A local plotter that updates a window in real-time.
        """
        # Enable interactive mode for live updates
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        
        # Store labels and limits
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        
        # Dictionary to store data for multiple lines: {'line_name': ([x_vals], [y_vals])}
        self.data = {}
        
    def draw(self, x, y, label, every_n=1):
        """
        x: current x value
        y: current y value
        label: name of the line (e.g., 'sin', 'cos')
        """
        # 1. Initialize data storage for this label if new
        if label not in self.data:
            self.data[label] = ([], [])
            
        # 2. Append new data
        xs, ys = self.data[label]
        xs.append(x)
        ys.append(y)
        
        # 3. Clear and Redraw
        # Note: Clearing and redrawing is inefficient for massive data but simple for learning.
        self.ax.clear()
        
        # Re-apply labels/limits after clear
        if self.xlabel: self.ax.set_xlabel(self.xlabel)
        if self.ylabel: self.ax.set_ylabel(self.ylabel)
        if self.xlim: self.ax.set_xlim(self.xlim)
        if self.ylim: self.ax.set_ylim(self.ylim)
        self.ax.grid(True)
        
        # Plot all lines stored in memory
        for name, (line_x, line_y) in self.data.items():
            self.ax.plot(line_x, line_y, label=name)
            
        self.ax.legend()
        
        # 4. Force the GUI to update
        plt.draw()
        plt.pause(0.01)  # Pause briefly to allow the window to render

# --- 2. Your Execution Code ---

# Create the board
board = LocalProgressBoard(xlabel='x', ylabel='y')

# Run the loop
print("Starting animation...")
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin')
    board.draw(x, np.cos(x), 'cos')
    
print("Animation finished. Close the window to exit.")
plt.ioff() # Turn off interactive mode
plt.show() # Keep window open at the end


### 3.2.2 MODELS ###

class Module(nn.Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, value.to(d2l.cpu()).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
    
    
### 3.2.3 DATA ###

class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    
    
### 3.2.4 TRAINING ###

class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError