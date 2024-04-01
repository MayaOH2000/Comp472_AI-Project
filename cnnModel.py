#Contains the actual cnn model and training process

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from skorch import NeuralNetClassifier
from torch.utils.data import DataLoader, Subset, random_split
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets
import torch.optim as optim

#data directory for local computer for dataset
dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"

#path to save the model that is being train
modelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/Models/"

#Seting up the pretraining-process
#Hyperparameters
num_epochs = 10     #minimum of 10 iteration
num_classes = 4     #total number of classes
learningRate = 0.001          #learnin rate for the model
batchSize = 32

#transform
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),   #Images are in grayscaled
    transforms.Resize((48,48)),#image size 48 x 48
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

#getting dataset for model to train on
dataset = torchvision.datasets.ImageFolder(dataPath, transform=transform)

#randomely spliting the datset into training and testing (80% for training and 20 % for testing)
m = len(dataset)
trainData, testData = random_split(dataset, [int (m-m*0.2), int(m*0.2)])
trainLoader = DataLoader(trainData, batch_size=32, shuffle=True, num_workers=2)
testLoader = DataLoader(testData, batch_size=32, shuffle=True, num_workers=2)

#checking if user has cude to be able to use GPU instead of CPU for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

y_train = np.array ([y for x, y,in iter (trainData)])
classes = ("angry", "neutral", "engaged", "surpise")    #classes for classification

#Doc string used here for documentation purposes
"""++++++ CNN Class Starts here ++++++ 
Creating neural netwrok architecture here.
Data set image format for images is 48 x 48 grayscale (black and white).
Creating 2 pools 
"""
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

"""
Below is the training process for the CNN model.
Along with the showing of the loss for each epoch(iteration) for the model being trained.
Saving the model after the training been done and
Showing the accuracy of the model on the test dataset.
"""

#creating an instance of the cnn model created from above
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

#training modle
# total_step = len(trainLoader)
# loss_list = []
# acc_list = []

# for epoch in range(num_epochs):
#  for i, (images, labels) in enumerate(trainLoader):

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss_list.append(loss.item())

#         # Backprop and optimisation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Train accuracy
#         total = labels.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         correct = (predicted == labels).sum().item()
#         acc_list.append(correct / total)
            
#         print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
#             .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
#             (correct / total) * 100))
torch.manual_seed(0)
net = NeuralNetClassifier(
    CNN(),
    max_epochs=num_epochs,
    lr=learningRate,
    batch_size=batchSize,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=device,
    iterator_train__shuffle=True
)
y_train = np.array([np.int64(y) for x, y in iter(trainData)])
net.fit(trainData, y=y_train)
print("\nFinished Training!!")
        
#testing model that just been trained
y_predict = net.predict(testData)
y_test = np.array([y for x, y in iter(testData)])
print('Accuracy for {} Dataset: {}%'.format('Test', round(accuracy_score(y_test, y_predict) * 100, 2)))
ConfusionMatrixDisplay.from_predictions(y_test, y_predict, display_labels=classes)
plt.title('Confusion Matrix for {} Dataset'.format('Test'))
plt.show()


# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in testLoader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print('Test Accuracy of the model on the test images: {} %'
# .format((correct / total) * 100))

#saving the model
torch.save(model.state_dict(), modelPath)