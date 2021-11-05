# import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

import numpy as np
import torch
device=torch.device('cuda')
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataSet(Dataset):
    def __init__(self,filepath):
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.x_train=torch.from_numpy(data[:,:-1]).to(device)
        self.y_train=torch.from_numpy(data[:,[-1]]).to(device)
        self.size=data.shape[0]

    def __getitem__(self, item):
        return self.x_train[item],self.y_train[item]

    def __len__(self):
        return self.size

dataset=MyDataSet('./dataset/diabetes.csv.gz')
print('dataset prepared!')
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=1)


class MultiLayer(torch.nn.Module):
    def __init__(self):
        super(MultiLayer, self).__init__()
        self.first=torch.nn.Linear(8,6)
        self.second=torch.nn.Linear(6,3)
        self.third=torch.nn.Linear(3,1)
        self.activate=torch.nn.Sigmoid()

    def forward(self,x):
        x=self.activate( self.first(x))
        x=self.activate( self.second(x))
        y=self.activate( self.third(x))
        return y

model=MultiLayer().to(device)

criterion=torch.nn.BCELoss(size_average=True).to(device)
optim=torch.optim.SGD(model.parameters(),lr=0.001)

for epoch in range(1000):
    for i,batch in enumerate(train_loader,0):
        x_train,y_train=batch
        # x_train=x_train
        # y_train=y_train
        y_pred=model(x_train)
        loss=criterion(y_pred,y_train)
        print(epoch,i,loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
