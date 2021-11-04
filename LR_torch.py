import  torch
import numpy
device=torch.device('cuda:0')
# 尝试使用中文注释
# prepare the data set
x_train=torch.Tensor([[1.],[2.],[3.]])
y_train=torch.Tensor([[2.],[4,],[6.]])
x_train=x_train.to(device)
y_train=y_train.to(device)

# establish the node model
class LinearModel(torch.nn.Linear):
    def __init__(self):
        super(torch.nn.Linear,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred

unit=LinearModel()
unit=unit.to(device)
# criterion and optim
criterion=torch.nn.MSELoss(True,True).to(device)
optim=torch.optim.Adam(unit.parameters(),0.01)

# training cycle
for epoch in range(1000):
    y_pred=unit(x_train)
    loss=criterion(y_pred,y_train)
    print(epoch,loss)
    optim.zero_grad()
    loss.backward()
    optim.step()

# output the results
print('w:',unit.linear.weight.data)
print('b:',unit.linear.bias.data)

# prediction
x_test=torch.Tensor([[4.]]).to(device)
y_test=unit(x_test)
print('y_test:',y_test.data)
