import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

data = pd.read_csv('data_set_2d_non_linear.csv')

x_train = torch.tensor(data["day"].values, dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor(data["head circumference"].values, dtype=torch.float).reshape(-1, 1)

# Observed/training input and output

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20*torch.sigmoid(x @ self.W + self.b) + 31  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)
for epoch in range(100000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))

# Visualize result
plt.title('Head circumference based on nr. of days')
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$', markersize=2)
plt.xlabel('days')
plt.ylabel('head circumference')
x = torch.arange(torch.min(x_train), torch.max(x_train), 1.0).reshape(-1, 1)
y = model.f(x).detach()

plt.plot(x, y, color='orange', linewidth=3,
         label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$')

plt.legend()
plt.show()
