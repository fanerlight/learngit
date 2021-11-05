'''
Inception是一个封装好的并行卷积神经网络结构，其灵感是使用并行不同网络结构来尝试得到数个网络中较优的一条分支
concatenate是将卷积层在通道维度进行拼接
1*1卷积核的作用：直接改变输入输出层的通道数；使用1*1卷积核可以降低运算复杂度。未知数的个数会变小
'''
import torch
device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

batch_size=32
transformer=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307,0.3081)
])
# preparation of dataset
train_set=datasets.MNIST('./dataset/MNIST',train=True,transform=transformer)
train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,num_workers=3)
test_set=datasets.MNIST('./dataset/MNIST',train=False,transform=transformer)
test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False,num_workers=3)

'''
以四条并行分支的Inception为例，一条分支是将输入层进行平均池化；另外一条并行的层是1x1卷积改变层数；
第三条并行线为3x3卷积与池化；第四条并行线是5x5卷积与池化。
需要注意的地方是最后四条并行线路得到的输出层长宽应该是相同的。
为了增强该类的复用性，给入一个构造参数——inchannel表示输入层的通道数量
'''
class Inception(torch.nn.Module):
    # 为什么这里要传入输入层数？一会视频复习考虑。
    '''
    传入输入层数的目的，是为了让这个Inception模块能够在不同的地方被复用。
    如果in_channel被写死，意味着整个这个Inception结构只能用于某些特定的输出C
    '''
    def __init__(self,in_channel):
        super().__init__()
        # 1x1卷积核大小已经确定了输出的大小与原图像相同，所以其他并行分支也必须要得到与原图相同的大小输出
        self.branch_1x1=torch.nn.Conv2d(in_channel,out_channels=24,kernel_size=1)
        # 池化卷积层为了获得与输入层相同的长宽高，在池化的时候不能是默认stride为2，而应该成为1，并且kernel也不再是2
        self.branch_pool_1=torch.nn.AvgPool2d(kernel_size=3,padding=1,stride=1)
        self.branch_pool_2=torch.nn.Conv2d(in_channel,out_channels=24,kernel_size=1)
        # 5*5大小的卷积核，为了得到W、H不变的输出，padding应该设置为2
        self.branch_5x5_1=torch.nn.Conv2d(in_channel,out_channels=16,kernel_size=5,padding=2)
        self.branch_5x5_2=torch.nn.Conv2d(in_channels=16,out_channels=24,kernel_size=5,padding=2)
        # 3*3大小的卷积核，为了得到与输入W、H相同的初始，padding应该设置为1
        self.branch_3x3_1=torch.nn.Conv2d(in_channel,out_channels=16,kernel_size=3,padding=1)
        self.branch_3x3_2=torch.nn.Conv2d(in_channels=16,out_channels=24,kernel_size=3,padding=1)
        self.branch_3x3_3=torch.nn.Conv2d(in_channels=24,out_channels=24,kernel_size=3,padding=1)

    def forward(self,x:torch.Tensor):
        branch_pool=self.branch_pool_2(self.branch_pool_1(x))
        branch_1x1=self.branch_1x1(x)
        branch_3x3=self.branch_3x3_1(x)
        branch_3x3=self.branch_3x3_2(branch_3x3)
        branch_3x3=self.branch_3x3_3(branch_3x3)
        branch_5x5=self.branch_5x5_1(x)
        branch_5x5=self.branch_5x5_2(branch_5x5)
        return torch.cat([branch_pool,branch_1x1,branch_5x5,branch_3x3],dim=1)

'''
下面是一个实际在训练MNIST中使用的网络，其中包含有上面构造的四分支inception，卷积之后用两个全连接层得到分类
'''
class Net(torch.nn.Module):
    # 输入是batch、1、28、28
    def __init__(self,):
        super().__init__()
        self.incep1=Inception(1)# 96*28*28
        self.conv=torch.nn.Conv2d(96,20,5)# 20*24*24
        self.pool=torch.nn.MaxPool2d(2)#20*12*12
        self.activate=torch.nn.ReLU()
        self.incep2=Inception(20) # inception并不改变长宽，到这里96*144
        self.fc=torch.nn.Linear(13824,1024)
        self.fc2=torch.nn.Linear(1024,256)
        self.fc3=torch.nn.Linear(256,10)

    def forward(self,x:torch.Tensor):
        batch=x.size(0)
        x= self.activate(self.incep1(x))
        x=self.activate(self.pool(self.conv(x)))
        x=self.incep2(x)
        x=x.view(batch,-1)
        x=self.activate(self.fc(x))
        x=self.activate(self.fc2(x))
        x=self.fc3(x)
        return x

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
        test()
