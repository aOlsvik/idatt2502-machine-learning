import torch
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data_set_2d_linear.csv')

x_train = torch.tensor(data[data.columns[0]].values, dtype=torch.double).reshape(-1, 1)
y_train = torch.tensor(data[data.columns[1]].values, dtype=torch.double).reshape(-1, 1)

# Observed/training input and output

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True, dtype=torch.double)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.double)

    # Predictor
    def f(self, x, y):
        return x, y @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(500000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$', markersize=1)
plt.xlabel('length')
plt.ylabel('weight')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$f(x) = xW+b$')
plt.legend()
plt.show()
