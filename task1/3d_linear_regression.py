import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data_set_3d_linear.csv')

x_data = data[['length', 'weight']].values
y_data = data['day'].values.reshape(-1, 1)

x_train = torch.tensor(x_data, dtype=torch.float)
y_train = torch.tensor(y_data, dtype=torch.float)

# Observed/training input and output
class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True, dtype=torch.float)
        self.b = torch.tensor([0.0], requires_grad=True, dtype=torch.float)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.00001)
for epoch in range(500000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b.item(), model.loss(x_train, y_train).item()))

xt =x_train.t()[0]
yt =x_train.t()[1]


xt = x_train.t()[0]
yt = x_train.t()[1]

# Visualize result
fig = plt.figure("Linear regression 3D")
ax = fig.add_subplot(projection='3d', title="Model for predicting days lived by weight and length")
ax.set_xlabel("length")
ax.set_ylabel("weight")
ax.set_zlabel("day")

# Scatter plot of the training data
ax.scatter(xt.numpy(), yt.numpy(), y_train.numpy(), label='$(x^{(i)},y^{(i)}, z^{(i)})$')

# Generate a meshgrid of x and y values
x_range = np.linspace(xt.min(), xt.max(), 10)
y_range = np.linspace(yt.min(), yt.max(), 10)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Create a 2D array of input features from the meshgrid
input_features = torch.tensor(np.vstack((x_grid.flatten(), y_grid.flatten())), dtype=torch.float).t()

# Predict z values using the model
z_pred = model.f(input_features).detach().numpy()

# Reshape the predicted values to match the shape of the meshgrid
z_pred = z_pred.reshape(x_grid.shape)

# Plot the predicted plane using plot_surface
ax.plot_surface(x_grid, y_grid, z_pred, color='red', alpha=0.5, label='$\\hat y = f(x) = xW+b$')


plt.show()

