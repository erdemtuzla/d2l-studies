import torch

print("---LINEAR ALGEBRA---")


print("## SCALARS ##")
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y)
print(x * y)
print(x / y)
print(x ** y)

print("## VECTORS ##")
x = torch.arange(3)
print(x)

print(x[2])

print(len(x))

print(x.shape)

print("## MATRICES ##")
A = torch.arange(6).reshape(3, 2)
print(A)

# Transpose
print(A.T)

# Symmetric Matrices
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(A == A.T)

print("## TENSORS ##")
A = torch.arange(24).reshape(2, 3, 4)
print(A)

print("## BASIC PROPERTIES OF TENSOR ARITHMETIC")

A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()
print(A + B)


# Elementwise product (Hadamart Product)
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)

print(a + X)
print((a * X).shape)

print("## REDUCTION ##")

x = torch.arange(3, dtype=torch.float32)
print(x)
print(x.sum())

print(A)
print(A.sum())
print(A.sum(axis=0))
print(A.sum(axis=1))

# Mean
print(A.mean())

print("## NN=ON REDUCTION SUMS ##")
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

print("## DOT PRODUCT ##")
y = torch.ones(3, dtype=torch.float32)
print(torch.dot(x, y))
print(torch.sum(x * y))

print("## MATRIX-VECTOR PRODUCTS ##")
print(A)
print(x)
print(A.shape, x.shape, torch.mv(A, x))

print("## MATRIX-MATRIX MULTIPLICATION ##")
B = torch.ones(3, 4)
print(A)
print(B)
print(torch.mm(A, B))

print("## NORM ##")
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))



# END
print()