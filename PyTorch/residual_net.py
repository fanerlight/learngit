import torch
device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')
'''
主要是为了解决梯度消失问题，提出了跳连接的概念。它将输入层直接与串行经过的输出层进行相加作为结果层（可以激活）
整体这样一个模块称之为残差网络（跳连接网络）
'''
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
transformer=transforms.Compose([ #compose的组合，表示将多个变换组合成为一个pipeline
    transforms.ToTensor(),# 首先进行张量化
    transforms.Normalize((0.1307, ),(0.3081, )) # 接着通过标准化使数据符合标准正态分布
])
batch_size=64


# preparation of dataset
train_set=datasets.MNIST('./dataset/MNIST',train=True,transform=transformer)
train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,num_workers=3)
test_set=datasets.MNIST('./dataset/MNIST',train=False,transform=transformer)
test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False,num_workers=3)


'''
首先创建可复用的残差层。残差层的限定条件就是最终输出一定要和输入在cwh三个维度上都相等
设我们使用两个卷积层作为残差层的运算
'''
class Residual(torch.nn.Module):
    def __init__(self,in_channels):# 残差网络的超参数应该是什么？
        super().__init__()
        self.activate=torch.nn.ReLU()
        self.conv1=torch.nn.Conv2d(in_channels=in_channels,out_channels=30,kernel_size=3,padding=1)
        self.conv2=torch.nn.Conv2d(in_channels=30,out_channels=15,padding=2,kernel_size=5)
        self.conv3=torch.nn.Conv2d(in_channels=15,out_channels=in_channels,kernel_size=1)

    def forward(self,x:torch.Tensor):
        x1=self.activate(self.conv1(x))
        x1=self.activate(self.conv2(x1))
        x1=self.activate(self.conv3(x1))
        return self.activate(x+x1)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activate=torch.nn.ReLU()
        self.pooling=torch.nn.MaxPool2d(2)
        self.conv1=torch.nn.Conv2d(1,20,5)#20*24*24,pooling 20*12*12
        self.Rd1=Residual(20)
        self.conv2=torch.nn.Conv2d(20,10,5)# 10*8*8,pooling 10*4*4
        self.Rd2=Residual(10)
        self.fc1=torch.nn.Linear(in_features=160,out_features=80)
        self.fc2=torch.nn.Linear(80,10)
    
    def forward(self,x:torch.Tensor):
        batch=x.size(0)
        x=self.activate(self.pooling(self.conv1(x)))
        x=self.Rd1(x)
        x=self.activate(self.pooling(self.conv2(x)))
        x=self.Rd2(x)
        x=x.view(batch,-1)
        x=self.activate(self.fc1(x))
        return self.fc2(x)

model=Net().to(device)
criterion=torch.nn.CrossEntropyLoss().to(device)
optim=torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.5)

def train(current_epoch):
    total_loss=0.
    for batch_idx,data in enumerate(train_loader,0):
        input,label=data
        input,label=input.to(device),label.to(device)
        optim.zero_grad()
        output=model(input)
        # print(output.shape)
        # break
        loss=criterion(output,label)
        loss.backward()
        optim.step()
        total_loss+=loss.item()
        # break
        if batch_idx%300==299:
            print('[%d %d] loss:%.5f'%(current_epoch+1,batch_idx+1,total_loss))

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            input,label=data
            input,label=input.to(device),label.to(device)
            output=model(input)
            _,predict=torch.max(output,dim=1)
            correct+=(predict==label).sum().item()
            total+=label.size(0)
    print('Accuracy: %.3f %% [%d/%d]'%(100.*float(correct)/float(total),correct,total))

if __name__=='__main__':
    for i in range(10):
        train(i)
        # break
        test()



