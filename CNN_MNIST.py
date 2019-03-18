import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import math
import statistics

# モデルを用意
class _CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.l1 = nn.Linear(16*5*5, 120)
        self.l2 = nn.Linear(120, 10)

    def forward(self, x):
        # (batch_size, 1, 28, 28) -> (batch_size, 16, 13, 13)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        # (batch_size, 16, 13, 13) -> (batch_size, 16, 5, 5)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        # (batch_size, 16, 5, 5) -> (batch_size, 10)
        x = x.view(-1, 16*5*5)
        x = self.l1(x)
        x = self.l2(x)
        return x

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.l1 = nn.Linear(16*5*5, 120)
        self.l2 = nn.Linear(120, 10)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # (batch_size, 1, 28, 28) -> (batch_size, 16, 13, 13)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.pool(x)

        # (batch_size, 16, 15, 15) -> (batch_size, 16, 5, 5)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.pool(x)

        # (batch_size, 16, 5, 5) -> (batch_size, 10)
        x = x.view(-1, 16*5*5)
        x = self.l1(x)
        x = F.leaky_relu(x)
        x = self.l2(x)
        return x

# モデルのイニシャライズ
model = CNN_model()

# 訓練データとテストデータを用意
train_data = MNIST('~/tmp/mnist',
                  train=True, download=True,
                  transform=transforms.ToTensor())
train_loader = DataLoader(train_data,
                         batch_size=16,
                         shuffle=True)
test_data = MNIST('~/tmp/mnist',
                 train=False, download=True,
                 transform=transforms.ToTensor())
test_loader = DataLoader(test_data,
                         batch_size=16,
                         shuffle=False)

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
