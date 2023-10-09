import torch
import matplotlib.pyplot as plt
import numpy as np

train_x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).reshape(-1,2)
train_y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

W1_init = torch.tensor([[10.0, -10.0], [10.0, -10.0]], requires_grad=True)
b1_init =  torch.tensor([[-5.0, 15.0]], requires_grad=True)
W2_init =  torch.tensor([[10.0], [10.0]], requires_grad=True)
b2_init =  torch.tensor([[-15.0]], requires_grad=True)

class XOROperatorModel:
    def __init__(self, W1=W1_init, W2=W2_init, b1=b1_init, b2=b2_init):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    # Predictor
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    def f2(self,x):
        return torch.sigmoid(x @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))  
    

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x),y)
    
model = XOROperatorModel()

model2 = XOROperatorModel(
    torch.rand((2,2), requires_grad=True),
    torch.rand((2,1), requires_grad=True),
    torch.rand((1,2), requires_grad=True),
    torch.rand((1,1), requires_grad=True)
)

optimizer = torch.optim.SGD([model.b1, model.W1, model.W2, model.b2], lr=0.1)
optimizer2 = torch.optim.SGD([model2.b1, model2.W1, model2.W2, model2.b2], lr=0.1)
for epoch in range(50000):
    model.loss(train_x, train_y).backward()  
    optimizer.step() 
    optimizer.zero_grad()  
    model2.loss(train_x, train_y).backward()  
    optimizer2.step() 
    optimizer2.zero_grad()  


print("Model 1 W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" %(model.b1, model.W1, model.W2, model.b2, model.loss(train_x, train_y)))
print("Model 2W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" %(model2.b1, model2.W1, model2.W2, model2.b2, model2.loss(train_x, train_y)))

xt =train_x.t()[0]
yt =train_x.t()[1]

fig = plt.figure("Logistic regression: the logical OR operator")

plot1 = fig.add_subplot(111, projection='3d')

plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$\\hat y=f(\\mathbf{x})=\\sigma(\\mathbf{xW}+b)$")

plot1.plot(xt.squeeze(), yt.squeeze(), train_y[:, 0].squeeze(), 'o', label="$(x_1^{(i)}, x_2^{(i)},y^{(i)})$", color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$y$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

table = plt.table(cellText=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
                  colWidths=[0.1] * 3,
                  colLabels=["$x_1$", "$x_2$", "$f(\\mathbf{x})$"],
                  cellLoc="center",
                  loc="lower right")


plot1_f.remove()
def plotModel(model, plot, color):
    x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
    y_grid = np.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f(torch.tensor([[(x1_grid[i, j]),  (x2_grid[i, j])]], dtype=torch.float))
    plot1_f = plot.plot_wireframe(x1_grid, x2_grid, y_grid, color=color)

plotModel(model, plot1, "green")
plotModel(model2, plot1, "orange")

fig.canvas.draw()

plt.show()