import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


x_train = torch.tensor([[0, 1], [0, 0], [1, 0], [1,1]], dtype=torch.float).reshape(-1,2)
y_train = torch.tensor([[1], [1], [1], [0]], dtype=torch.float)
# Observed/training input and output

class SigmoidModel:
    def __init__(self):
        self.W = torch.rand((2,1), requires_grad=True)
        self.b = torch.rand((1,1), requires_grad=True)
    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))
    
    def logits(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x),y)

model = SigmoidModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in range(50_000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b.item(), model.loss(x_train, y_train).item()))

# Visualize result
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
x1, x2 = np.meshgrid(x1, x2)
x_grid = np.column_stack((x1.ravel(), x2.ravel()))

# Calculate model predictions on the grid
y_grid = model.f(torch.tensor(x_grid, dtype=torch.float32)).detach().numpy()
y_grid = y_grid.reshape(x1.shape)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the surface representing the decision boundary
ax.plot_surface(x1, x2, y_grid, cmap='summer_r', alpha=0.8)
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c=y_train.ravel())
ax.set_xlabel('First input')
ax.set_ylabel('Second input')
ax.set_zlabel('Predicted Output')
plt.title('NAND operator')
plt.show()


