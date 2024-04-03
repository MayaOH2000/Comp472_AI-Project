#Contains the actual cnn model and training process

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix, pair_confusion_matrix, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from skorch.callbacks import EarlyStopping, Checkpoint
from skorch import NeuralNetClassifier
from torch.utils.data import DataLoader, random_split
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets
import torch.optim as optim
from cnnModel import CNN 
from cnnModel import CNNV1
from cnnModel import CNNV2


#data directory for local computer for dataset
dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"

#path to save the model that is being train
modelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv2"
 
#Set random seed to be the same each time to help with reproducability
torch.manual_seed(0)
np.random.seed(0)  

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
trainData, testData = random_split(dataset, [(m-int(m*0.2)), int(m*0.2)])

#checking if user has cude to be able to use GPU instead of CPU for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yTrain = np.array ([y for x, y,in iter (trainData)])
classes = ("angry", "neutral", "engaged", "surpise")    #classes for classification

"""
Below is the training process for the CNN model.
Along with the showing of the loss for each epoch(iteration) for the model being trained.
Saving the model after the training been done and
Showing the accuracy of the model on the test dataset.
"""

#creating an instance of the cnn model created from above
#model = CNNV1() #train with variant 1
model = CNNV2() #train with variant 2
#model = CNN()   #train with main
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

#Early stop implementation
earlyStop = EarlyStopping(monitor='valid_loss', patience=3, lower_is_better=True)

#Best fit model 
bestFit= Checkpoint(monitor='valod_loss_best', fn_prefix="best_")

#training modle
torch.manual_seed(0)
net = NeuralNetClassifier(
    model,
    max_epochs=num_epochs,
    lr=learningRate,
    batch_size=batchSize,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=device
)

#Tested with train data
print(trainData.dataset)
print(yTrain.shape)

#model fitting
net.fit(trainData, y=yTrain)
print("\nFinished Training!!")


#evaluationmof train model
y_predict = net.predict(testData)
y_test = np.array([y for x, y in iter(testData)])

accuracy_score(y_test, y_predict)
print('Accuracy for {} Dataset: {}%'.format('Test', round(accuracy_score(y_test, y_predict) * 100, 2)))

#Printing report out for recall,precision, f1-score and accuracy of model
print(classification_report(y_test, y_predict))

#saving the model
torch.save(model.state_dict(), modelPath) 

#svaing best model
best_model = modelPath + "_best.pth"
torch.save(model.state_dict(), best_model)

#confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_predict, display_labels=classes)
plt.title('Confusion Matrix for {} Dataset'.format('Test'))
plt.show()



