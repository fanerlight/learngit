import torch
device=torch.device('cuda')
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

train_dataset=torchvision.datasets.MNIST('./dataset/MNIST',True,transform=torchvision.transforms.ToTensor(),target_transform=torchvision.transforms.ToTensor())
test_dataset=torchvision.datasets.MNIST('./dataset/MNIST',False,transform=torchvision.transforms.ToTensor())

train=DataLoader(train_dataset,32,True,num_workers=12)
test=DataLoader(test_dataset,32,False,num_workers=12)

class LRModel(torch.nn.Module):
    def __init__(self):
        super(LRModel, self).__init__()
        self.unit=torch.nn.Linear(784,1)
        self.activate=torch.nn.Sigmoid()

    def forward(self,x):
        z_pred=self.activate(self.unit(x))
        return z_pred

unit=LRModel().to(device)

critetion=torch.nn.BCELoss(reduction='mean').to(device)
optim=torch.optim.Adam(unit.parameters(),lr=0.001)


for epoch in range(100):
    for i,[x_train,z_train] in enumerate(train,0):
        x_train=x_train.to(device).view([32,784])
        z_train=z_train.to(device).view([32,1])
        z_pred=unit(x_train)
        loss=critetion(z_pred,z_train)
        print(epoch,i,loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
