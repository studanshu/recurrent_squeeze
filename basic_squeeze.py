
# coding: utf-8

# In[25]:

from __future__ import unicode_literals, print_function, division, absolute_import
import numpy as np
from io import open
import unicodedata, string, re, random, math, time, shutil, pdb, argparse, os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

begin = time.time()

use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
DEBUG = False
criterion = nn.CrossEntropyLoss()
print ("CUDA available ?", use_cuda)

# please make this change
load_model = False
#offset = 1200

#general constants
learning_rate=0.00001
momentum = 0.9
start_epoch = 0
num_epochs= 5000
num_classes=10
train_batch_size=128
test_batch_size=100
display_step=100
save_fig_step = 10
lr_decay_step = 2500
criterion = nn.CrossEntropyLoss()

#Debugging schemes.
train_accuracies_arr = np.zeros(num_epochs)
test_accuracies_arr = np.zeros(num_epochs)
train_loss_arr = np.zeros(num_epochs)
test_loss_arr = np.zeros(num_epochs)

curr_best_test = 0


# ## SqueezeNet Code - Pytorch

# In[26]:


import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2), # Using 1 for MNIST.
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2), # Using 1 for MNIST.
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(1, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return F.log_softmax(x.view(x.size(0), self.num_classes))
    
    def name():
        return "basicSqueezeNet"


def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model


# ## Loading the CIFAR-10 Dataset

# In[27]:


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = dsets.CIFAR10(root='./data-cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)

test_set = dsets.CIFAR10(root='./data-cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ## Loading the MNIST Dataset

# In[28]:


# train_loader = torch.utils.data.DataLoader(
#     dsets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#         batch_size=64, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     dsets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#         batch_size=1000, shuffle=True)


# In[29]:


print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


# In[30]:


def save_checkpoint(state, is_best, filename='models/basicsqueezenet.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'models/model_best_basic.tar')


# In[31]:


def train_epoch(epoch, model, data_loader, optimizer):
    global train_loss_arr, train_accuracies_arr
    # model.train()
    pid = os.getpid()
    train_loss = 0.
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        time_start = time.time()
        data, target = Variable(data.type(dtype)), Variable(target.type(dtype))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        train_loss += loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.long().data).cpu().sum()
        loss.backward()
        optimizer.step()
        #if batch_idx % display_step == 0:
            #print('Process id : {}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                #100. * batch_idx / len(data_loader), loss.data[0]))
            #print ("Time elapsed in batch number ", batch_idx," ",time.time() - time_start)
    
    train_loss /= len(data_loader.dataset)
    train_accuracy = (100. * correct) / len(data_loader.dataset)
    
    try:
        train_loss_arr[epoch] = train_loss
        train_accuracies_arr[epoch] = train_accuracy
    except:
        train_loss_arr[epoch] = train_loss.data[0]
        train_accuracies_arr[epoch] = train_accuracy
    
    #print("Here", train_accuracy, train_accuracies_arr[epoch])
    print('\nTrain set: Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))


# In[32]:


def test_epoch(epoch, model, data_loader):
    global curr_best_test, test_loss_arr, test_accuracies_arr
    # model.eval()
    test_loss = 0.
    correct = 0
    for data, target in data_loader:
        data, target = Variable(data.type(dtype), volatile=True), Variable(target.type(dtype))
        output = model(data)
        #test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        test_loss += criterion(output, target.long())
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.long().data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    test_accuracy = (100. * correct) / len(data_loader.dataset)
    
    try:
        test_loss_arr[epoch] = test_loss
        test_accuracies_arr[epoch] = test_accuracy
    except:
        test_loss_arr[epoch] = test_loss.data[0]
        test_accuracies_arr[epoch] = test_accuracy
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss.data[0], correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    if (100. * correct / len(data_loader.dataset)) > curr_best_test:
        curr_best_test = (100. * correct / len(data_loader.dataset))
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_loss': test_loss,
                'optimizer' : optimizer.state_dict(),
            }, True)


# In[ ]:


#Load saved Model
squeezenet = SqueezeNet(num_classes=num_classes)
#print ("..test log..", load_model)
if load_model:
    state=torch.load("models/model_best_basic.tar")
    squeezenet.load_state_dict(state['state_dict'])
    start_epoch = state['epoch']
    print (start_epoch)
squeezenet.train()


# In[ ]:


curr_best_test = 0
if use_cuda:
    squeezenet = squeezenet.cuda()
optimizer = optim.Adam(squeezenet.parameters(), lr=learning_rate)
#print (num_epochs)
for epoch in range(start_epoch, num_epochs):
    #if epoch < 2000: continue
    time_epoch_start = time.time()
    train_epoch(epoch, squeezenet, train_loader, optimizer)
    test_epoch(epoch, squeezenet, test_loader)
    
    print ("Time elapsed in EPOCH (seconds) ", epoch, " ", time.time() - time_epoch_start)
    print ("Time elapsed since the beginning (HH:MM:SS) ", time.strftime("%H:%M:%S", time.gmtime(time.time() - begin)) )
    if epoch % save_fig_step == 0 and epoch > 0:
        #Loss.
        plt.plot(range(start_epoch, epoch),test_loss_arr[start_epoch:epoch],'r--',range(start_epoch, epoch),train_loss_arr[start_epoch:epoch],'g--',label='red->test\ngreen->train')
        plt.legend()
        plt.savefig('Basic_loss_Adam.jpg')
        plt.clf()
        #Accuracies.
        plt.plot(range(start_epoch, epoch),test_accuracies_arr[start_epoch:epoch],'r--',range(start_epoch, epoch),train_accuracies_arr[start_epoch:epoch],'g--',label='red->test\ngreen->train')
        plt.legend()
        plt.savefig('Basic_accuracy_Adam.jpg')
        plt.clf()

    if epoch % lr_decay_step == 0 and epoch > 0:
        learning_rate = learning_rate * 1/2
        print ("learning_rate updated from {} to {}".format(learning_rate*2, learning_rate))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate



