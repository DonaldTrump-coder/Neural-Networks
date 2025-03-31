import torch.nn as nn
import torch
import torch.utils.data.dataloader
from torchvision import transforms
import tqdm

import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

import torch.utils
import torch.utils.data
import torch.optim as optim

#读取数据
class MnistDataloader():
    def __init__(self, 
                 training_images_filepath,
                 training_labels_filepath,
                 test_images_filepath, 
                 test_labels_filepath
                 ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        train_result=[]
        test_result=[]
        for i in range(len(x_train)):
            train_result.append(np.array(x_train[i]))
        for i in range(len(x_test)):
            test_result.append(np.array(x_test[i]))
        return (np.array(train_result), y_train),(np.array(test_result), y_test)

class CNNDataset(torch.utils.data.Dataset):
    def __init__(self,images:np.array,labels:np.array):
        super().__init__()
        images=torch.tensor(images,dtype=torch.float32)
        self.labels=torch.tensor(labels,dtype=torch.long)
        self.images=images.unsqueeze(1)/255
        self.transform=transforms.Compose([transforms.Normalize((0.5,),(0.5,))])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image=self.images[index]
        label=self.labels[index]
        image=self.transform(image)
        return image,label

batch=32
channel=1
width=28
height=28
classes=10
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class simpleCNN(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.features=nn.Sequential(#提取特征
            nn.Conv2d(channel,#输入通道数
                      16,#输出通道数
                      kernel_size=3,#卷积核大小
                      stride=1,#卷积核步长
                      padding=1
                      )#保持图像大小不变
                      ,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),#池化，图像大小减半
            nn.Conv2d(16,
                      32,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),#池化，图像大小减半
        )
        self.classifier=nn.Sequential(
            nn.Linear(32*int(width/4)*int(height/4),128),
            nn.ReLU(),
            nn.Linear(128,num_class)
        )
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
    

dataloader=MnistDataloader("/media/allen/新加卷/Convolutional Neural Network/data/train-images.idx3-ubyte","/media/allen/新加卷/Convolutional Neural Network/data/train-labels.idx1-ubyte","/media/allen/新加卷/Convolutional Neural Network/data/t10k-images.idx3-ubyte","/media/allen/新加卷/Convolutional Neural Network/data/t10k-labels.idx1-ubyte")
(x_train, y_train), (x_test, y_test) = dataloader.load_data()
train_dataset=CNNDataset(x_train,y_train)
test_dataset=CNNDataset(x_test,y_test)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch,shuffle=True)
#准备数据集，两个可迭代对象

def train(model:simpleCNN,train_loader,criterion,optimizer,epochs):
    for epoch in range(epochs):
        model.train()
        running_loss=0
        for inputs,labels in tqdm.tqdm(train_loader,desc=f"epoch:{epoch+1}/{epochs}",unit="batch"):#打印进度条
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()#梯度清零
            outputs=model(inputs)#前馈
            loss=criterion(outputs,labels)#计算损失函数
            loss.backward()#反向传播
            optimizer.step()#更新参数
            running_loss+=loss.item()*(inputs.size(0))
        epoch_loss=running_loss/len(train_loader.dataset)
        print(f"[epoch{epoch+1}/{epochs},Train_loss:{epoch_loss:.4f}]")

def evaluate(model:simpleCNN,test_loader,criterion):
    model.eval()
    test_loss=0#测试中的损失
    correct=0#正确的样本数量
    total=0#总样本数量
    with torch.no_grad():
        for inputs,labels in tqdm.tqdm(test_loader,desc="",unit="batch"):#打印进度条
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs,labels)#计算损失函数
            test_loss+=loss.item()*(inputs.size(0))
            total+=labels.size(0)
            _,predicted=torch.max(outputs,1)
            correct+=(predicted==labels).sum().item()
    
    avg_loss=test_loss/len(test_loader.dataset)
    acc=100.0*correct/total
    print(f"Test Loss:{avg_loss:.4f},Accuracy:{acc:.2f}%")

epochs=10
learning_rate=0.001
model=simpleCNN(classes).to(device)
criterion=nn.CrossEntropyLoss()#交叉熵损失函数
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
train(model,train_loader,criterion,optimizer,epochs)
evaluate(model,test_loader,criterion)