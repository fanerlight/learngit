'''
感受野穿透上一层输出的全部层；
卷积核层数和上一层输出的层数相同，并且共享权重。每层卷积做完之后，对应位置相加得到输出。
卷积层包含输入通道数量、输出通道数量以及卷积核大小。这些参数决定了卷积层的未知数个数。torch.nn.Conv2d
其他与卷积核相关的参数：
    padding：填充输入图像的边缘，使得输出的大小发生变化
    stride：确定卷积核移动的个数（左右移动和上下移动）
'''
import torch
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(device)
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

# model establishing
class MNISTConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling=torch.nn.MaxPool2d(2)
        self.activation=torch.nn.ReLU()
        # 原始输入28*28
        self.kernel1=torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3)
        # 26*26，池化13*13
        self.kernel2=torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)
        # 11*11
        self.kernel3=torch.nn.Conv2d(in_channels=20,out_channels=5,kernel_size=5,stride=2,padding=1)
        # 4*4，池化2*2，一共5层
        self.fc=torch.nn.Linear(in_features=20,out_features=10)
    
    def forward(self,x:torch.Tensor):# 这个进来应该是以batch为组的，不是单个样本，所以会有batch这个维度
        batch=x.size(0)
        x=self.pooling(self.kernel1(x))
        x=self.activation(x)
        # print(x.size())
        x=self.activation(self.kernel2(x))
        # print(x.size())
        x=self.pooling(self.kernel3(x))
        x=self.activation(x)# relu和pooling的先后其实没有关系，不影响
        # print(x.size())
        x=x.view(batch,-1)# 做全连接的时候需要把他压平
        return x

model=MNISTConv().to(device)

# loss and optimization
criterion=torch.nn.CrossEntropyLoss().to(device)
optim=torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.5)

# training cycle
def train(epoch):
    total_loss=0.
    for batch_idx,data in enumerate(train_loader,0):
        input,labels=data
        input,labels=input.to(device),labels.to(device)
        z=model(input)
        optim.zero_grad()
        loss=criterion(z,labels)
        loss.backward()
        optim.step()
        total_loss+=loss.item()
        if batch_idx%300 ==299:
            print('[%d %d]: %.3f'%(epoch+1,batch_idx+1,total_loss))

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            input,labels=data
            input,labels=input.to(device),labels.to(device)
            z=model(input)
            _,predicted=torch.max(z,dim=1)
            correct+= (labels==predicted).sum().item()
            total+=labels.size(0)
    print('Accuracy is %.3f %%'%(float(100*correct)/float(total)))

for i in range(10):
    train(i)
    test()

