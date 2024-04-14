#Contains the actual cnn model and training process
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from skorch.callbacks import EarlyStopping, Checkpoint
from skorch import NeuralNetClassifier
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets
import numpy as np
import torch.optim as optim

from cnnModel import CNN 
from cnnModel import CNNV1
from cnnModel import CNNV2


#data directory for local computer for dataset
dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"

#path to save the model that is being train
modelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model"    #main
#modelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv1"   #V1
#modelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv2"   #V2

if __name__ == "__main__": 
    #Set random seed to be the same each time to help with reproducability
    torch.manual_seed(0)
    np.random.seed(0)  

    #Seting up the pretraining-process
    #Hyperparameters
    num_epochs = 10     #minimum of 10 iteration
    num_classes = 4     #total number of classes
    learningRate = 0.001          #learnin rate for the model

    #transform
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),   #Images are in grayscaled
        transforms.Resize((48,48)),#image size 48 x 48
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    #getting dataset for model to train on
    dataset = torchvision.datasets.ImageFolder(dataPath, transform=transform)

    #randomely spliting the datset into training and testing (70% for training, 20 % for testing and 10% for validation)
    m = len(dataset)
    trainSize = int(0.7*m)
    testSize = int(0.1*m)
    valSize = m - trainSize - testSize
    trainData, testData, valData = random_split(dataset, [trainSize,testSize,valSize])

    #Data Loader
    #allow random order for loading data (shuffle = true) and use 2 subprocess to load data
    trainLoader = DataLoader(trainData, batch_size=32, shuffle=True, num_workers=2)
    testLoader = DataLoader(testData, batch_size=32, shuffle=False, num_workers=2)
    valLoader = DataLoader(valData, batch_size=32, shuffle=False, num_workers=2)

    #checking if user has cuda to be able to use GPU instead of CPU for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #yTrain = np.array ([y for x, y,in iter (trainData)])
    classes = ("angry", "neutral", "engaged", "surpise")    #classes for classification

    """
    Below is the training process for the CNN model.
    Along with the showing of the loss for each epoch(iteration) for the model being trained.
    Saving the model after the training been done and
    Showing the accuracy of the model on the test dataset.
    """

    #creating an instance of the cnn model created from above
    #model = CNNV1() #train with variant 1
    #model = CNNV2() #train with variant 2
    model = CNN()   #train with main

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

    #Early stopping variables
    bestValLoss = float('inf')
    patience = 5
    noImprovementCount = 0

    #Best fit variables
    #best-fit save path
    bestModel = modelPath + "_best.pth"

    #training model
    #training loop
    for epoch in range(num_epochs):
        #training phase
        model.train()  
        trainCorrect = 0
        trainTotal = 0
        totalLoss = 0
        trainLoss = 0.0
        for i, (images,labels) in enumerate(trainLoader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #train loss
            trainLoss += loss.item() * images.size(0)

            #train accuarcy
            _, trainPredicted = torch.max(outputs.data, 1)
            trainTotal += labels.size(0)
            trainCorrect += (trainPredicted == labels).sum().item()


         #training accuracy and loss for each epoch
        trainAccuracy = trainCorrect/trainTotal
        trainLoss /= len(trainLoader.dataset)

        #validating train model and save best fit model
        model.eval()
        valLoss = 0.0
        valCorrect = 0
        valTotal = 0

        for images, labels in valLoader:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

                #val loss
                valLoss += loss.item() * images.size(0)

                #val accuracy
                _, predicted = torch.max(outputs.data, 1)
                valTotal += labels.size(0)
                valCorrect += (predicted == labels).sum().item()

        #validation results for each epoch
        valLoss /= len(valLoader.dataset)
        valAccuracy = valCorrect/valTotal

        #print results for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, "
              f"Training Accuracy: {trainAccuracy * 100:.2f}%, Validation Accuracy: {valAccuracy*100:.2f}%")
        
        #best fit checking
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            #save best-fit model
            torch.save(model.state_dict(), bestModel)
            noImprovementCount = 0
        else:
            noImprovementCount += 1

        #Early stopping
        if noImprovementCount > patience:
            print("*** Early Stopping Happened! ****")
            break

    #saving the final model
    torch.save(model.state_dict(), modelPath)
     
    print("\n ++++++ Training Complete!! +++++ ")



