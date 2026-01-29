import time
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt

# Sets the display format to SVG (Scalable Vector Graphics) for sharper images in notebooks.
d2l.use_svg_display()

# ==========================================
# 1. DEFINING THE DATASET CLASS
# ==========================================

# We inherit from d2l.DataModule, which is a wrapper around PyTorch's data handling.
# It helps organize data splitting (train/val) and loading logic in one place.
class FashionMNIST(d2l.DataModule):
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        # save_hyperparameters() automatically saves __init__ arguments (batch_size, resize)
        # into 'self' (e.g., self.batch_size). Helpful for debugging and cleaner code.
        self.save_hyperparameters()
        
        # TRANSFORMS: A pipeline of operations applied to images before they enter the model.
        # 1. Resize: Changes image resolution. (32, 32) is common for models like LeNet/ResNet
        #    to ensure dimensionality matches standard architectures.
        # 2. ToTensor: Critical step. Converts PIL images (integers 0-255, shape HxWxC) 
        #    into PyTorch Tensors (floats 0.0-1.0, shape CxHxW).
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        
        # Load the Training Set
        # root=self.root: Where to save data on disk (usually '../data').
        # train=True: Loads the 60,000 training images.
        # download=True: Downloads from the internet if not found locally.
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=True,
            transform=trans,
            download=True
        )
        
        # Load the Validation Set
        # train=False: Loads the 10,000 test/validation images.
        # We use this to check model performance on unseen data.
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=False,
            transform=trans,
            download=True
        )
        
# Initialize the class with a resize to 32x32 pixels.
data = FashionMNIST(resize=(32, 32))

print(f'Training data count: {len(data.train)} \nValidation data count: {len(data.val)}')

# Check the shape of a single image.
# Output is typically [1, 32, 32] -> [Channels, Height, Width].
# "1" channel means it is Grayscale. RGB would be "3".
print(f'{data.train[0][0].shape}')

# Use a decorator to add this method to the existing FashionMNIST class.
# This maps the numeric class ID (e.g., 9) to its human-readable name (e.g., 'ankle boot').
@d2l.add_to_class(FashionMNIST)
def text_labels(self, indices):
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]

# ==========================================
# 2. READING DATA IN MINIBATCHES
# ==========================================

@d2l.add_to_class(FashionMNIST)
def get_dataloader(self, train):
    # Select the appropriate dataset (train vs validation)
    data = self.train if train else self.val
    
    # DataLoader: The engine that feeds data to the model.
    # batch_size: How many images to process at once (e.g., 64). 
    #             Larger batches use more VRAM but stabilize gradients.
    # shuffle=True: Critical for training. Randomizes order so the model doesn't 
    #               memorize the sequence of images (prevents overfitting).
    # num_workers: Parallel processes for loading data from disk to RAM. 
    #              Prevents the GPU from waiting on the CPU to load files.
    return torch.utils.data.DataLoader(data, 
                                       self.batch_size, 
                                       shuffle=train,
                                       num_workers=self.num_workers)
    
# Extract one batch to verify shapes.
# iter(): Creates an iterator object.
# next(): Grabs the first batch from that iterator.
X, y = next(iter(data.train_dataloader()))

# X shape: [64, 1, 32, 32] -> [Batch Size, Channels, Height, Width]
# y shape: [64] -> Vector of 64 integer labels (0-9)
print(f'X shape: {X.shape} \ny shape: {y.shape}')
print(f'X dtype: {X.dtype} \ny dtype: {y.dtype}')

# Performance Check: Measure how long it takes to iterate through the whole dataset.
tic = time.time()
for X, y in data.train_dataloader():
    continue # Pass does nothing; we just want to measure loading speed.
print(f'{time.time() - tic:.2f} seconds elapsed')

# ==========================================
# 3. VISUALIZATION
# ==========================================

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    # Calculate total figure size based on number of columns/rows and scale factor.
    figsize = (num_cols * scale, num_rows * scale)
    
    # Create subplots grid.
    # _: We ignore the Figure object.
    # axes: Array of Axes objects (the actual plots).
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Flatten the 2D grid of axes (rows x cols) into a 1D array for easy looping.
    axes = axes.flatten()
    
    # zip(axes, imgs): Pairs each plot axis with an image tensor.
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image: Check if it's a PyTorch tensor.
            # .numpy(): Convert tensor to numpy array because Matplotlib requires numpy.
            ax.imshow(img.numpy())
        else:
            # PIL Image: If it's already a standard image format, just show it.
            ax.imshow(img)
            
        # Hide the X and Y axis ticks (numbers) for a cleaner look.
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        # If labels were provided, set the title above the image.
        if titles:
            ax.set_title(titles[i])
            
    return axes

@d2l.add_to_class(FashionMNIST)
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    # Unpack the batch into Inputs (X) and Labels (y)
    X, y = batch
    
    # If no custom labels provided, convert numeric labels (y) to text (e.g., 0 -> 't-shirt')
    if not labels:
        labels = self.text_labels(y)
        
    # X.squeeze(1): Removes the channel dimension.
    # Input X is [Batch, 1, 32, 32]. 
    # Matplotlib grayscale expects [Height, Width] or [Height, Width, Channel].
    # Squeeze turns [Batch, 1, 32, 32] into [Batch, 32, 32], which works for plotting.
    show_images(X.squeeze(1), nrows, ncols, titles=labels)
    
# Get a batch from validation set and visualize it.
batch = next(iter(data.val_dataloader()))
data.visualize(batch)

# Explicitly show the plot window (required in scripts outside Jupyter notebooks).
plt.show()