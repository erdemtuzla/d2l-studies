import torch

x = torch.arange(12, dtype=torch.float32)
X = x.reshape(3, 4)


print(x)
print(x.numel())
print(x.shape)

print("----------")


print(X)
print(X.numel())
print(X.shape)

print("----------")

zeros = torch.zeros(2, 3, 4)

print(zeros)
print(zeros.numel())
print(zeros.shape)

print("----------")

ones = torch.ones(2, 3, 4)

print(ones)
print(ones.numel())
print(ones.shape)

print("----------") 

rand = torch.randn(3, 4)

print(rand)
print(rand.numel())
print(rand.shape)

print("----------")

tensor = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(tensor)
print(tensor.numel())
print(tensor.shape)

print("----------")
print(":")

print(X[-1])

print(X[1:3])

print("----------")

X[1, 2] = 17

print(X)

print("----------")

X[:2, :] = 12

print(X)

print("----------")

print(torch.exp(x))

print("----------")

X1 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
X2 = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(torch.cat((X1, X2), dim=0), torch.cat((X1, X2), dim=1))

print("----------")

print(X1 == X2)

print("----------")

print(X.sum())

print("----------")
print("BROADCASTING")

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

print(a)
print(b)
print(a + b)

print("----------")
print("CONVERSION TO OTHER PYTHON OBJECT")

A = X.numpy()
B = torch.from_numpy(A)

print(type(A))
print(type(B))


a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))