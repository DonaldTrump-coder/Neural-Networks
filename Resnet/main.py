from torchvision import transforms
from dataloader import MyDataset
from torch.utils.data import DataLoader
from resnet import Resnet,BasicBlock,Bottleneck
import torch
import torch.nn as nn
import torch.optim as optim

def resnet18(num_class=1000,include_top=True,pretrained=False):
    return Resnet(BasicBlock,[2,2,2,2],num_classes=num_class,include_top=include_top)

def resnet34(num_class=1000,include_top=True,pretrained=False):
    return Resnet(BasicBlock,[3,4,6,3],num_classes=num_class,include_top=include_top)

def resnet50(num_class=1000,include_top=True,pretrained=False):
    return Resnet(Bottleneck,[3,4,6,3],num_classes=num_class,include_top=include_top)

def resnet152(num_class=1000,include_top=True,pretrained=False):
    return Resnet(Bottleneck,[3,4,23,3],num_classes=num_class,include_top=include_top)

transform = transforms.Compose([
    #transforms.Resize((64,64)),  # 可根据模型需要修改尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,], std=[0.5,])
])#将图形从PIL格式转变为张量

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
traindata=MyDataset("data/train-00000-of-00001-1359597a978bc4fa.parquet",transform)
num_classes=traindata.get_num_classes()
train_dataloader = DataLoader(traindata, batch_size=32, shuffle=True)
model=resnet34(num_class=num_classes)
model = model.to(device)
epochs=10

valdata=MyDataset("data/valid-00000-of-00001-70d52db3c749a935.parquet",transform)
val_dataloader = DataLoader(valdata, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images,labels in val_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()*(images.size(0))
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    train_acc=correct/total
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    print(f"Val Acc: {val_acc:.4f}")