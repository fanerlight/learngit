# oftmax is not an activate function, instead it regulates the sum of every label's probability equals 1.
# Particularly, the output of the softmax is frac{exp(z_i)}{sigma(exp(z))}. This ensures every label bigger than 0.
# the workflow of softmax layer follows: 1. calculate the exponent of every z;2. sum and divide.
import numpy as np

y=np.array([1,0,0])
z=np.array([0.2,0.2,-0.1])
y_pred=np.exp(z)/np.exp(z).sum()
loss=-(y*np.log(y_pred)).sum()
import  torch
device=torch.device('cuda')
criterion=torch.nn.CrossEntropyLoss().to(device)
Y=torch.LongTensor([2,0,1]).to(device)
Y_pred1=torch.Tensor([
    [0.1,0.2,0.9],[1.1,0.1,0.2],[0.2,2.1,0.1]
]).to(device)
Y_pred2=torch.Tensor([
    [0.8,0.2,0.3],[0.2,0.3,0.5],[0.2,0.2,0.5]
]).to(device)

print(criterion(Y_pred1,Y).data)
print(criterion(Y_pred2,Y).data)