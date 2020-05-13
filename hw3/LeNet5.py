#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# In[4]:


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# In[5]:

# In[8]:


# 训练过程
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        if (batch_idx+1)%30 == 0:
            print("Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
            with open('./lenet5_train_log.txt', 'a+') as f:
                    f.write("Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}\n".format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))

    with SummaryWriter('./lenet_scalar') as writer:#自动调用close()
        writer.add_scalar('lenet_scalar/train_loss', running_loss/len(train_loader.dataset), epoch)
        writer.add_scalar('lenet_scalar/train_accuracy', eval(model, device, train_loader, criterion), epoch)
        writer.add_scalar('lenet_scalar/test_accuracy', eval(model, device, test_loader, criterion, is_train=False), epoch)
# In[9]:


# 评估函数
def eval(model, device, test_loader, criterion, is_train=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # 获得最大概率下标
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    if is_train:
        print('\nTrain Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f})%\n'.format(
            test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))
        with open('./lenet5_train_log.txt', 'a+') as f:
            f.write('\nTrain Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f})%\n'.format(
            test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))
    else:
        print('\nValid Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f})%\n'.format(
            test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))
        with open('./lenet5_train_log.txt', 'a+') as f:
            f.write('\nValid Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f})%\n'.format(
            test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))
    return 100.*correct/len(test_loader.dataset)


# In[ ]:
if __name__ =='__main__':
    batch_size=512
    epoch=20
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # In[6]:


    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True,
                                                                     transform=transforms.Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.1307,), (0.3081,)),
                                                                     ])),
                                              batch_size=batch_size,
                                              shuffle=True)


    # In[7]:


    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False,
                                                                     transform=transforms.Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                                     ])),
                                              batch_size=batch_size,
                                              shuffle=False)

    # 模型的损失函数与优化器
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    for ep in range(1, epoch+1):
        train(model, device, train_loader, criterion, optimizer, ep)


    # In[11]:


    torch.save(model, './LetNet5_model.pth')

