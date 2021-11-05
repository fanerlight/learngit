import torch
device=torch.device('cuda')
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

'------数据集准备-------'
batch_size=64
'''
从PIL（python image library）的得到的每个像素是[0,255]之间的整数，通道的排列是W*H*C；
但是神经网络希望输入的数据比较小，如[0,1]之间的实数；并且pytorch会将多通道图像的维度变为C*W*H
torchvision的transforms模块提供了将数据集变换为torch可用的Tensor数据。
'''
transforms=transforms.Compose([ #compose的组合，表示将多个变换组合成为一个pipeline
    transforms.ToTensor(),# 首先进行张量化
    transforms.Normalize((0.1307, ),(0.3081, )) # 接着通过标准化使数据符合标准正态分布
])
'''
DataLoader需要来自Datasets的配合。首先从Dataset库中读取数据，需要路径和变换对象
Set给入Loader之后，由Loader进行batch的分配和顺序打乱。然后使用enumerate枚举其中索引和数据项
每个数据项，其实就是一个batch，他在原来张量数据的基础上再次增加一个维度，该维度的size是batch_size
'''
train_data=datasets.MNIST('./dataset/MNIST',True,transform=transforms)
train_loader=DataLoader(train_data,batch_size,True)
test_data=datasets.MNIST('./dataset/MNIST',False,transform=transforms)
test_loader=DataLoader(test_data,batch_size,False)

'-------网络模型建立----------'
class MultiLayer(torch.nn.Module):
    def __init__(self):
        super(MultiLayer, self).__init__()
        self.l1=torch.nn.Linear(784,512)
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,64)
        self.l4=torch.nn.Linear(64,10)

    def forward(self,x):
        # 每个样本其实是一个高维张量，维度是1*28*28。但是线性层要求输入为一个向量，所以需要用到view变换形状
        x=x.view(-1,784)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=torch.relu(self.l3(x))
        # x=torch.nn.Softmax(self.l4(x)) # 教程中，这里最后一层没有使用softmax？这个和使用的损失函数有关
        return self.l4(x)

model=MultiLayer().to(device)
'''
---------损失函数和优化器确定-----------
这里使用了交叉熵损失函数，要求的输入并不需要做softmax和log。
'''
criterion=torch.nn.CrossEntropyLoss().to(device)
optim=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

'--------训练循环和测试（Training Cycle）----------'
def train(epoch):
    running_loss=0.
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        inputs=inputs.to(device)
        target=target.to(device)
        optim.zero_grad() # 先进行梯队归零
        y_pred=model(inputs)
        loss=criterion(y_pred,target)
        loss.backward()
        optim.step()

        running_loss+=loss.item()
        if batch_idx%300==299:
            print('[%d,%d] loss: %.3f'% (epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.

def test():
    correct=0.
    total=0.
    with torch.no_grad():# 不需要求loss的反向传播梯度，节省资源
        for data in test_loader:
            images,labels=data
            images=images.to(device)
            labels=labels.to(device)
            z=model(images)
            _,predict=torch.max(z.data,1)# max得到一个张量在dim维度上的最大值，返回value本身以及所在索引
            total+=labels.size(0)
            correct+=(predict==labels).sum().item()
    print("Accuracy on test set:%.3f %%" % (float(100)*correct/total))


if __name__=='__main__':
    '将batch训练封装起来，一共训练epoch轮，训练完之后就进行测试'
    for epoch in range(10):
        train(epoch)
        test()