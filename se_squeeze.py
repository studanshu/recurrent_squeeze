
# coding: utf-8

# In[144]:


from __future__ import unicode_literals, print_function, division, absolute_import
import numpy as np
from io import open
import unicodedata, string, re, random, math, time, shutil, pdb, argparse, os
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

begin = time.time()
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

DEBUG = False
print ("CUDA available ?", use_cuda)

load_model = False

#general constants
learning_rate=0.00001
momentum = 0.9
start_epoch = 0
num_epochs = 5000
num_classes=10
train_batch_size=128
test_batch_size=100
display_step=100
save_fig_step=10
lr_decay_step=2000
criterion = nn.CrossEntropyLoss()

#lstm constants
lstm_input_size=128
lstm_hidden_size=128

#Debugging schemes.
train_accuracies_arr = np.zeros(num_epochs)
test_accuracies_arr = np.zeros(num_epochs)
train_loss_arr = np.zeros(num_epochs)
test_loss_arr = np.zeros(num_epochs)

curr_best_test = 0


# # The LSTM cell module

# In[145]:


class LSTMCellModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCellModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
    
    def forward(self, x):
        try:
            self.hidden, self.cell_state = self.lstm_cell(x, (self.hidden, self.cell_state))
        except:
            pdb.set_trace()
        return self.hidden
    
    def initHidden(self, batch_sz):
        global use_cuda
        self.hidden = Variable(torch.zeros(batch_sz, self.hidden_size))
        self.cell_state = Variable(torch.zeros(batch_sz, self.hidden_size))
        if use_cuda:
            self.hidden = self.hidden.cuda()
            self.cell_state = self.cell_state.cuda()


# ## Recurrent Attention Module

# In[146]:


class RecurrentAttention(nn.Module):
    def __init__(self, num_channel, lstm_cell):
        super(RecurrentAttention, self).__init__()
        self.linear_input_lstm = nn.Linear(num_channel, lstm_cell.input_size)
        self.lstm_cell = lstm_cell
        self.linear_output_lstm = nn.Linear(lstm_cell.hidden_size, num_channel)
    
    def forward(self, x):
        x_sz = x.size()
#         pdb.set_trace()
        self.pool_layer = nn.AvgPool2d(kernel_size=(x_sz[2], x_sz[3]), stride=1)
        return             F.sigmoid(
                self.linear_output_lstm(
                    #self.lstm_cell(
                        F.relu(
                            self.linear_input_lstm(
                                torch.squeeze(
                                    self.pool_layer(x)
                                )
                            )
                        )
                    #)
                )
            ).unsqueeze(2).unsqueeze(3).expand_as(x)*x


# ## SqueezeNet Code - Pytorch (with Recurrent Attention)

# In[147]:


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


class RecurrentFire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, lstm_cell_unit):
        super(RecurrentFire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.recurrent_attention_squeeze = RecurrentAttention(squeeze_planes, lstm_cell_unit)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.recurrent_attention_expand = RecurrentAttention(expand1x1_planes + expand3x3_planes, lstm_cell_unit)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.recurrent_attention_squeeze(self.squeeze(x)))
        out_expand1x1 = self.expand1x1(x)
        out_expand3x3 = self.expand3x3(x)
        out_expand_cat = torch.cat((out_expand1x1, out_expand3x3),1)
        out_rnn_attn = self.recurrent_attention_expand(out_expand_cat)
        return self.expand_activation(out_rnn_attn)


class RecurrentSqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(RecurrentSqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        global lstm_input_size, lstm_hidden_size
        self.lstm_cell_unit = LSTMCellModule(lstm_input_size, lstm_hidden_size)
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2), # Using 1 for MNIST.
                RecurrentAttention(96, self.lstm_cell_unit),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                RecurrentFire(96, 16, 64, 64, self.lstm_cell_unit),
                RecurrentFire(128, 16, 64, 64, self.lstm_cell_unit),
                RecurrentFire(128, 32, 128, 128, self.lstm_cell_unit),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                RecurrentFire(256, 32, 128, 128, self.lstm_cell_unit),
                RecurrentFire(256, 48, 192, 192,self.lstm_cell_unit),
                RecurrentFire(384, 48, 192, 192,self.lstm_cell_unit),
                RecurrentFire(384, 64, 256, 256,self.lstm_cell_unit),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                RecurrentFire(512, 64, 256, 256,self.lstm_cell_unit),
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
            nn.AvgPool2d(1, stride=1) #Changed for MNIST.
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

    def initHidden(self, batch_sz):
        self.lstm_cell_unit.initHidden(batch_sz)
    
    def name():
        return "recurrentSqueezeNet"


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

# In[148]:


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

# In[149]:


# train_loader = torch.utils.data.DataLoader(
#     dsets.MNIST('../data-mnist', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#         batch_size=train_batch_size, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     dsets.MNIST('../data-mnist', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#         batch_size=test_batch_size, shuffle=True)


# In[150]:


print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


# In[151]:


def save_checkpoint(state, is_best, filename='models/recurrentsqueezenet.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'models/model_best_se.tar')


# In[152]:


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
        model.initHidden(data.size()[0])
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
    train_accuracy = 100. * correct / len(data_loader.dataset)
    
    try:
        train_loss_arr[epoch] = train_loss
        train_accuracies_arr[epoch] = train_accuracy
    except:
        train_loss_arr[epoch] = train_loss.data[0]
        train_accuracies_arr[epoch] = train_accuracy
    
    print('\nTrain set: Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))


# In[153]:


def test_epoch(epoch, model, data_loader):
    global curr_best_test, test_loss_arr, test_accuracies_arr
    # model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        data, target = Variable(data.type(dtype), volatile=True), Variable(target.type(dtype))
        model.initHidden(data.size()[0])
        output = model(data)
        #test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        test_loss += criterion(output, target.long())
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.long().data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    test_accuracy = 100. * correct / len(data_loader.dataset)
    
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
squeezenet = RecurrentSqueezeNet(1.0, num_classes)
if load_model:
    state=torch.load("models/model_best_se.tar")
    squeezenet.load_state_dict(state['state_dict'])
    start_epoch = state['epoch']
    print (start_epoch)
squeezenet.train()


# In[ ]:


curr_best_test = 0
if use_cuda:
    squeezenet = squeezenet.cuda()
optimizer = optim.Adam(squeezenet.parameters(), lr=learning_rate)
for epoch in range(start_epoch, num_epochs):
#     if epoch < 299: continue
    time_epoch_start = time.time()
    train_epoch(epoch, squeezenet, train_loader, optimizer)
    test_epoch(epoch, squeezenet, test_loader)
    
    print ("Time elapsed in EPOCH ", epoch, " ", time.time() - time_epoch_start) 
    print ("Time elapsed since the beginning (HH:MM:SS) ", time.strftime("%H:%M:%S", time.gmtime(time.time() - begin)) )
    
    if epoch % save_fig_step == 0 and epoch > 0:
        #Loss.
        plt.plot(range(start_epoch, epoch),test_loss_arr[start_epoch:epoch],'r--',range(start_epoch, epoch),train_loss_arr[start_epoch:epoch],'g--',label='red->test\ngreen->train')
        plt.legend()
        plt.savefig('SE_Attention_loss.jpg')
        plt.clf()
        #Accuracies.
        plt.plot(range(start_epoch, epoch),test_accuracies_arr[start_epoch:epoch],'r--',range(start_epoch, epoch),train_accuracies_arr[start_epoch:epoch],'g--',label='red->test\ngreen->train')
        plt.legend()
        plt.savefig('SE_Attention_accuracy.jpg')
        plt.clf()
    
    if epoch % lr_decay_step == 0 and epoch > 0:
        learning_rate = learning_rate * 1/2
        print ("learning_rate updated from {} to {}".format(learning_rate*2, learning_rate))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate



