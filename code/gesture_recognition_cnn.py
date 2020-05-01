import numpy as np
import pandas as pd
import torch.nn.functional as F
import math
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
import itertools
from torchvision import models
import torch.optim as optim
from matplotlib.ticker import MaxNLocator
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from torch.nn import MaxPool2d
import chainer.links as L
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
from torchsummary import summary
import pdb
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device == torch.device('cuda'):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

# Data Loading and Preprocessing
#This data is wrongly matched. Please execute this code to have the correct mapping of X and y values
data = np.load('../dataset/X.npy')
target = np.load('../dataset/Y.npy')
Y = np.zeros(data.shape[0])
Y[:204] = 9
Y[204:409] = 0
Y[409:615] = 7
Y[615:822] = 6
Y[822:1028] = 1
Y[1028:1236] = 8
Y[1236:1443] = 4
Y[1443:1649] = 3
Y[1649:1855] = 2
Y[1855:] = 5
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size = .2, random_state = 2) ## splitting into train and test set

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

class DatasetProcessing(Dataset):
    
    #initialise the class variables - transform, data, target
    def __init__(self, data, target, transform=None): 
        self.transform = transform
        self.data = data.reshape((-1,64,64)).astype(np.float32)[:,:,:,None]
        # converting target to torch.LongTensor dtype
        self.target = torch.from_numpy(target).long() 
    
    #retrieve the X and y index value and return it
    def __getitem__(self, index): 
        return self.transform(self.data[index]), self.target[index]
    
    #returns the length of the data
    def __len__(self): 
        return len(list(self.data))

# preprocessing images and performing operations sequentially
# Firstly, data is converted to PILImage, Secondly, converted to Tensor
# Thirdly, data is Normalized
train_transform = transforms.Compose(
    [transforms.ToPILImage(), 
    # transforms.RandomHorizontalFlip(), # Horizontal Flip
    # transforms.RandomCrop(64, padding=2), # Centre Crop
    transforms.ToTensor()])

test_transform = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.ToTensor()])

dset_train = DatasetProcessing(X_train, y_train, train_transform)

train_loader = torch.utils.data.DataLoader(dset_train, batch_size=4,
                                          shuffle=True, num_workers=4, drop_last=True)

dset_test = DatasetProcessing(X_test, y_test, test_transform)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=4,
                                          shuffle=True, num_workers=4, drop_last=True)

class Net(nn.Module):    
    
    # This constructor will initialize the model architecture
    def __init__(self):
        super(Net, self).__init__()
          
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
          
        self.linear_layers = nn.Sequential(
            # Adding Dropout
            nn.Dropout(p = 0.5),
            nn.Linear(32 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 10),
        )
        
    # Defining the forward pass    
    def forward(self, x):
        
        # Forward Pass through the CNN Layers 
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # Forwrd pass through Fully Connected Layers
        x = self.linear_layers(x)
        return F.log_softmax(x) 

model = Net().to(device)
summary(model, input_size=(1, 64, 64))
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

########################################
#       Training the model             #
########################################
def train(epoch):
    model.train()
    exp_lr_scheduler.step()
    tr_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
            
        # Clearing the Gradients of the model parameters
        optimizer.zero_grad()
        output = model(data)
        pred = torch.max(output.data, 1)[1]
        correct += (pred == target).sum()
        total += len(data)
        
        # Computing the loss
        loss = criterion(output, target)
        
        # Computing the updated weights of all the model parameters
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        if (batch_idx + 1)% 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {} %'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(),100 * correct / total))
            torch.save(model.state_dict(), './model.pth')
            torch.save(model.state_dict(), './optimizer.pth')
    train_loss.append(tr_loss / len(train_loader))
    train_accuracy.append(100 * correct / total)

########################################
#       Evaluating the model           #
########################################

def evaluate(data_loader):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss += F.cross_entropy(output, target, size_average=False).item()
        pred = torch.max(output.data, 1)[1]
        total += len(data)
        correct += (pred == target).sum()
    loss /= len(data_loader.dataset)
    test_loss.append(loss)    
    test_accuracy.append(100 * correct / total)
    print('\nAverage test loss: {:.5f}\tAccuracy: {} %'.format(loss, 100 * correct / total))

n_epochs = 100
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []
for epoch in range(n_epochs):
    train(epoch)
    evaluate(test_loader)

########################################
#       Plotting the Graph             #
########################################

def plot_graph(epochs):
    fig = plt.figure(figsize=(20,4))
    ax = fig.add_subplot(1, 2, 1)
    plt.title("Train - test Loss")
    plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')
    plt.plot(list(np.arange(epochs) + 1), test_loss, label='test')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='best')
    
    ax = fig.add_subplot(1, 2, 2)
    plt.title("Train - test Accuracy")
    plt.plot(list(np.arange(epochs) + 1) , train_accuracy, label='train')
    plt.plot(list(np.arange(epochs) + 1), test_accuracy, label='test')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='best')
    fig.savefig('exp1/train_vs_test_plot.png')

plot_graph(n_epochs)