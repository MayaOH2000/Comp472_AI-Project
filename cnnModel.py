#Contains the actual cnn model and training process

from torch.utils.data import DataLoader, Subset, random_split
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

#data directory for local computer for dataset
dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"

#Seting up the pretraining-process
#Hyperparameters
num_epochs = 10     #minimum of 10 iteration
num_classes = 4     #total number of classes
lr = 0.001          #learnin rate for the model
classes = ("angry", "neutral", "engaged", "surpise")    #classes for classification

#transform
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#getting dataset for model to train on
dataset = torchvision.datasets.ImageFolder(dataPath, transform=transform)


#randomely spliting the datset into training, test and validation

#checking if user has cude to be able to use GPU instead of CPU for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#creating neural netwrok
#Image size is 48 x 48 grayscale (black and white)
#creating 2 pools 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),    # channel = 1 for gray scale
        nn.BatchNorm2d(32),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(12 * 12 * 64, 1000),  #48/2/2 = 12 
        nn.ReLU(inplace=True),
        nn.Linear(1000, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10)
        )
    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        #flatten
        x = x.view(x.size(0), -1)
        #fc layer
        x = self.fc_layer(x)
        return x
    
#training modle
    total_step = len(trainLoader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainLoader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
            .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
            (correct / total) * 100))