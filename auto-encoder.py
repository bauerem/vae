import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

height = 28 # MNIST: 28, CIFAR-10: 32
width = 28 # MNIST: 28, CIFAR-10: 32
channels = 1 # MNIST: 1, CIFAR-10: 3
batch_size = 4
k = 20 # Encoding size
kernel = 5
padding = 0
stride = 1
pool = 2
min_pixel = 0
max_pixel = 0

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(channels * (0.5,), channels * (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = tuple(range(10))

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{str(classes[labels[j]]):5s}' for j in range(batch_size)))


import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        w_o1 = int((width - kernel + 2*padding) / stride + 1)
        h_o1 = w_o1
        w_o2 = int(w_o1 / pool)
        h_o2 = w_o2
        w_o3 = int((w_o2 - kernel + 2*padding) / stride + 1)
        h_o3 = w_o3
        self.w_o4 = int(w_o3 / pool )
        self.h_o4 = self.w_o4
        
        # Encoding layers
        self.conv1 = nn.Conv2d(channels, 2*channels, kernel) # MNIST: (1, 2, 5), CIFAR-10: (3, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(2*channels, 6*channels, kernel) # MNIST: (2, 6, 5), CIFAR-10: (6, 18, 5)
        self.fc1 = nn.Linear(6*channels * self.w_o4 * self.h_o4, 40 * channels) # MNIST: (6*19*19, 40), CIFAR-10: (18*5*5, 120)
        self.fc2 = nn.Linear(40 * channels, 30 * channels) # MNIST: (40, 20), CIFAR-10: (120, 84)
        self.fc3 = nn.Linear(30 * channels, k) # MNIST: (20, 10), CIFAR-10: (84, 10)

        # Decoding layers
        self.fc4 = nn.Linear(k, 30 * channels)
        self.fc5 = nn.Linear(30 * channels, 40 * channels)
        self.fc6 = nn.Linear(40 * channels, 6*channels * self.w_o4 * self.h_o4 )
        self.deconv1 = nn.ConvTranspose2d(6*channels, 2*channels, kernel)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv2 = nn.ConvTranspose2d(2*channels, channels, kernel)


    def encode(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def decode(self,x):
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        x = x.reshape(batch_size,6*channels,self.h_o4, self.w_o4) # unflatten all dimensions except batch

        x = F.relu(self.deconv1(self.upsample(x)))
        x = F.relu(self.deconv2(self.upsample(x)))
        
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


net = AutoEncoder()

images = net(images)

images, labels = next(dataiter)

optimizer = torch.optim.Adam(net.parameters())
Loss = torch.nn.MSELoss()

def train(x,y,model,optimizer,Loss):
    optimizer.zero_grad()
    yhat = model(x)
    #yhat = model.encode(x)
    #y_ = torch.zeros(batch_size, len(classes))
    #y_[torch.arange(batch_size), y] = 1
    #y = y_
    loss = Loss(y,yhat)
    loss.backward()
    optimizer.step()
    return loss.detach()

losses = []
for i in tqdm(range(10000)):
    loss = train(images, images, net, optimizer, Loss)
    losses.append(loss)

images = net(images)

plt.figure()
plt.plot(np.log(losses))
plt.show()
print(losses[::100])

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{str(classes[labels[j]]):5s}' for j in range(batch_size)))