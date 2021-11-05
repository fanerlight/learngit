import torch
device=torch.device('cuda')
import torchvision
train_set=torchvision.datasets.MNIST(root='./dataset/MNIST',train=True,download=True)
test_set=torchvision.datasets.MNIST(root='./dataset/MNIST',train=False,download=True)

x_train=torch.Tensor([[1.],[2.],[3.]]).to(device)
y_train=torch.Tensor([[0],[0],[1]]).to(device)

class LRModel(torch.nn.Module):
    def __init__(self):
        super(LRModel, self).__init__()
        self.model=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=torch.sigmoid(self.model(x))
        return y_pred

unit=LRModel().to(device)
criterion=torch.nn.BCELoss().to(device)
optim=torch.optim.Adam(unit.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred=unit(x_train)
    loss=criterion(y_pred,y_train)
    print(epoch,loss)
    optim.zero_grad()
    loss.backward()
    optim.step()

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10,200)
x_t=torch.Tensor(x).view((200,1)).to(device)
y_t=unit(x_t)
y=y_t.data.cpu().numpy()
plt.plot(x,y)
plt.xlabel('Hours')
plt.ylabel('possibility of pass')
plt.grid()
plt.show()