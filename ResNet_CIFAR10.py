import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import math
import statistics

# モデルを用意
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # strideが1のとき、inputとoutputは同じ形
        # strideが2 or それ以上のとき、outputは小さくなる
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.conv_x = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.bn_x = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride != 1:
            x = self.conv_x(x)
            x = self.bn_x(x)
        out += x
        out = F.leaky_relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.res_layer1 = BasicBlock(16, 16)
        self.res_layer2 = BasicBlock(16, 16)
        self.res_layer3 = BasicBlock(16, 32, 2)
        self.res_layer4 = BasicBlock(32, 32)
        self.res_layer5 = BasicBlock(32, 32)
        self.res_layer6 = BasicBlock(32, 64, 2)
        self.res_layer7 = BasicBlock(64, 64)
        self.res_layer8 = BasicBlock(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.linear = nn.Linear(64*8*8, 10)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.res_layer1(out)
        out = self.res_layer2(out)
        out = self.res_layer3(out)
        out = self.res_layer4(out)
        out = self.res_layer5(out)
        out = self.res_layer6(out)
        out = self.res_layer7(out)
        out = self.res_layer8(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# モデルのイニシャライズ
model = ResNet()

# 訓練データとテストデータを用意
train_data = CIFAR10('~/tmp/cifar10', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_data = CIFAR10('~/tmp/cifar10', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# GPUの設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Lossの計算関数を設定
criterion = nn.CrossEntropyLoss()

# 値の更新関数を設定
optimizer = optim.Adam(model.parameters(), lr=0.01)

# test時に実行される関数
def test():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    # print(outputs, predicted, labels)
    print('Accuracy {} / {} = {}\n'
         .format(correct, total, float(correct) / total))

# 以下を実行しtrainingする
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = []
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 勾配情報をリセット
        optimizer.zero_grad()

        # 順伝播
        outputs = model(inputs)
        
        # コスト関数を使ってロスを計算する
        loss = criterion(outputs, labels)
        
        # 逆伝播
        loss.backward()
        
        # パラメータの更新
        optimizer.step()
        
        # ロスを保存
        running_loss.append(loss.item())
        
        # 中間結果を表示
        cut_number = 200
        if i % cut_number == 0:
            print('{} {} loss: {}'
                 .format(epoch + 1, i, statistics.mean(running_loss)))
            running_loss = []
    test()
            
print('Finished Training')
