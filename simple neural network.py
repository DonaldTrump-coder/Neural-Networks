import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np

x=np.linspace(-10,10,1000)
y=x**2

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        #三层网络
        self.layer1=nn.Linear(1,128)
        self.layer2=nn.Linear(128,128)
        self.layer3=nn.Linear(128,1)
    def forward(self,x):#前向计算
        x=torch.relu(self.layer1(x))
        x=torch.relu(self.layer2(x))
        x=self.layer3(x)
        return x
    def DataLoader(self,x,y):#张量数据加载
        self.X_train=torch.tensor(x,dtype=torch.float32).unsqueeze(1)
        self.Y_train=torch.tensor(y,dtype=torch.float32).unsqueeze(1)

model=SimpleNN()
criterion=nn.MSELoss()
model.DataLoader(x,y)#加载x，y数据
optimizer=optim.Adam(model.parameters(),#模型的参数，放入优化器中
                     lr=0.001#学习率
                     )
epochs=10000
losses=[]#存储每一步的损失值
for epoch in range(epochs):
    outputs=model(model.X_train)
    loss=criterion(outputs,model.Y_train)
    optimizer.zero_grad()#梯度清零
    loss.backward()#反向传播
    optimizer.step()#更新参数
    losses.append(loss.item())
    if(epoch+1)%100==0:
        print(f'Epoch[{epoch+1}/{epochs}],Loss:{loss.item():.4f}')

plt.plot(losses)
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

x=np.linspace(-100,100,1000)
y=x**2
model.DataLoader(x,y)
predicted=model(model.X_train).detach().numpy()#不再用于梯度计算

plt.scatter(x,y,label='Actual',color='blue',s=1)
plt.scatter(x,predicted,label='Predicted',color='red',s=1)
plt.legend()
plt.show()