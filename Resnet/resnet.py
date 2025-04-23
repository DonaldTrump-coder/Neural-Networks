import torch.nn as nn
import torch

class BasicBlock(nn.Module):# 用于浅层的Resnet
    expansion=1 # 膨胀因子

    def __init__(self, in_channel,out_channel,stride=1,downsample=None):
        super().__init__()
        # 初始化要用到的网络层
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding=1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.downsample=downsample # 下采样策略
    
    def forward(self,x):
        identity=x # 保存输入，便于进行残差连接
        if self.downsample is not None:
            identity=self.downsample(x)

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x+=identity
        x=self.relu(x)

        return x
    
class Bottleneck(nn.Module):
    expansion=4 # 最后一层的卷积核个数会变为第一层的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.conv3=nn.Conv2d(in_channels=out_channel, out_channels=self.expansion * out_channel, kernel_size=1, stride=stride, bias=False)
        self.bn3=nn.BatchNorm2d(self.expansion * out_channel)
        self.relu=nn.ReLU()
        self.downsample=downsample
    
    def forward(self,x):
        identity=x

        if self.downsample is not None:
            identity=self.downsample(x)

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x+=identity
        x=self.relu(x)

        return x
    
class Resnet(nn.Module):
    def __init__(self,
                 block,#block选取类型
                 blocks_num,#block的数量
                 num_classes,#分类数
                 include_top=True#分类头，为线性层
                 ):
        super().__init__()
        self.include_top=include_top
        self.in_channel=64
        self.conv1=nn.Conv2d(3,#输入为3通道
                             self.in_channel,
                             kernel_size=7,
                             stride=2,
                             bias=False,
                             padding=3)#会将图像大小减半
        self.bn1=nn.BatchNorm2d(self.in_channel)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer_(block,64,blocks_num[0])
        self.layer2=self._make_layer_(block,128,blocks_num[1],stride=2)
        self.layer3=self._make_layer_(block,256,blocks_num[2],stride=2)
        self.layer4=self._make_layer_(block,512,blocks_num[3],stride=2)
        if self.include_top:
            self.avgpool=nn.AdaptiveAvgPool2d((1,1))
            self.fc=nn.Linear(512*block.expansion,num_classes)

        #初始化卷积层的权重
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
    
    def _make_layer_(self,block,channel,block_num,stride=1):#创建残差块
        downsample=None
        if stride != 1 or self.in_channel != channel*block.expansion:
            downsample=nn.Sequential( #改变通道数
                nn.Conv2d(self.in_channel,channel*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )
        layers=[]
        layers.append(
            block(
                self.in_channel,
                channel,
                downsample=downsample,
                stride=stride
            )
        )
        self.in_channel=channel*block.expansion

        for _ in range(1,block_num):
            layers.append(
                block(
                    self.in_channel,
                    channel
                )
            )

        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        if self.include_top:
            x=self.avgpool(x)
            x=torch.flatten(x,1)#展平
            x=self.fc(x)
        
        return x