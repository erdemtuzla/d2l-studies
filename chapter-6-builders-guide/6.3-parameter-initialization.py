import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))

print(net(X).shape)

def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)
        
        
print("\n\nBefore[0]:")
print(net[0].weight.data[0])
print(net[0].bias.data[0])

print("Before[2]:")
print(net[2].weight.data[0])
print(net[2].bias.data[0])

net.apply(init_normal)

print("\n\nAfter[0]:")
print(net[0].weight.data[0])
print(net[0].bias.data[0])

print("After[2]:")
print(net[2].weight.data[0])
print(net[2].bias.data[0])

def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)

print("\n\nAfter init constant [0]:")
print(f'Weight: \n{net[0].weight}')
print(net[0].weight.data[0])
print(net[0].bias.data[0])

def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print("\n\nAfter init xavier and 42 [0]:")
print(net[0].weight.data[0])
print(net[2].weight.data)
print(net[2].bias)

def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)

print("\n\nAfter my init [0]:")
print(net[0].weight)


net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]

print(net[0].weight.data[0])
