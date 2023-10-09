import torch
import matplotlib.pyplot as plt
import pandas as pd

x_data = [[0.0], [1.0]]
y_data = [[1.0], [0.0]]

x_train = torch.tensor(x_data, dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor(y_data, dtype=torch.float).reshape(-1, 1)
# Observed/training input and output

class SigmoidModel:
    def __init__(self):
    # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b
    
    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))
    
    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)



model = SigmoidModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in range(100_000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))

# Visualize result
x_range = torch.arange(0, 1.0, 0.01).reshape(-1, 1)
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, label='Observations', s=150)
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_range, model.f(x_range).detach(), label='Regression model', color='red', linewidth=2)
plt.title('NOT operator')
plt.legend()
plt.grid(True)
plt.show()

