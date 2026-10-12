import numpy as np
import matplotlib.pyplot as plt
import torch


dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU

rng = np.random.default_rng()
n = 200
x = torch.linspace(0, 100, n, device=device, dtype=dtype)
a1 = 2
b1 = 5
#ecart-type
sig = 5
eps = rng.normal(0, sig, n)
i = 20000

# Create random input and output data
y = a1 * x + b1 + eps
y = torch.tensor(y, dtype=torch.float32)

# Randomly initialize weights
a2 = torch.randn((), device=device, dtype=dtype)
b2 = torch.randn((), device=device, dtype=dtype)


learning_rate = 1e-6
for t in range(i):
    # Forward pass: compute predicted y
    # y = a1*x + b1 + eps
    y_pred = a2*x + b2

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Backprop to compute gradients of a2, b2,  with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a2 = (grad_y_pred * x).sum()
    grad_b2 = (grad_y_pred ).sum()


    # Update weights
    a2 -= learning_rate * grad_a2
    b2 -= learning_rate * grad_b2
 
y_pred = a2*x + b2
print(f'Result: y = {a2.item()} x + {b2.item()}')

plt.close()
plt.grid()
plt.plot(x, y, 'o')
plt.plot(x,y_pred, 'b', label='y_pred')
plt.plot(x, a1*x + b1, 'r', label='y')
plt.show()