import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import math
import statistics

# モデルを用意
class NN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 50)
        self.l2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.l1(x)
        x = self.l2(x)
        return x

# モデルのイニシャライズ
model = NN_model()

# 訓練データとテストデータを用意
train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data,
                         batch_size=16,
                         shuffle=True)
test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())
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
    print('Accuracy {} / {} = {}' .format(correct, total, float(correct) / total))

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
            print('{} {} loss: {}'.format(epoch + 1, i, statistics.mean(running_loss)))
            running_loss = []
    test()
            
print('Finished Training')
