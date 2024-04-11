from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, make_scorer, precision_score,recall_score,f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from skorch import NeuralNetClassifier
from torch.utils.data import random_split
from skorch.helper import SliceDataset
from sklearn.model_selection import KFold

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets
import torch.optim as optim

from cnnModel import CNN
from cnnModel import CNNV1
from cnnModel import CNNV2

#change path to match data path and model to evaluate
loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model"   #main model
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv1"   #V1
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv2"   #V2

dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"


"""Evaluation with Confusion Matrix"""
def evaluate(cnnModel, testData,trainData, classes ):  
    torch.manual_seed(0)
    net = NeuralNetClassifier(
        cnnModel,
        max_epochs=10,
        lr=0.001,
        batch_size=32,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        verbose=0,  #supress training process output
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    )

    #Evaluation with Test Data
    net.fit(trainData, y=yTrain)
    
    #Evaluating model
    y_predict = net.predict(testData)
    y_test = np.array([y for x, y in iter(testData)])

    accuracy_score(y_test, y_predict)
    print('Accuracy for {} Dataset: {}%'.format('Test', round(accuracy_score(y_test, y_predict) * 100, 2)))

    #Printing report out for recall,precision, f1-score and accuracy of model
    print(classification_report(y_test, y_predict))

    #confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_predict, display_labels=classes)
    plt.title('Confusion Matrix for {} Dataset'.format('Test'))
    plt.show()
 
"""K-fold Cross-Validate with k = 10 folds"""
#Takes a long time to compute
def kFoldCrossValidation(cnnModel,trainData,yTrain,k = 10):
    torch.manual_seed(0)
    net = NeuralNetClassifier(
        cnnModel,
        max_epochs=10,
        lr=0.001,
        batch_size=32,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        #verbose=0, #supress training process output
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    )
    
    #net.fit(trainData, y=yTrain)
    yTrain = torch.tensor(yTrain).long()
    train_sliceable = SliceDataset(trainData)

    kf= KFold(n_splits=k, shuffle=True, random_state=0)

    #dictonary to store score values for k-fold values 
    #macro average
    scoringMacro = {"accuracy": make_scorer(accuracy_score),
               "precision": make_scorer(precision_score, average="macro") ,
               "recall": make_scorer(recall_score, average="macro"),
               "f1": make_scorer(f1_score, average="macro"),
               }
    #micro averaging
    scoringMicro = {"accuracy": make_scorer(accuracy_score),
               "precision": make_scorer(precision_score, average="micro") ,
               "recall": make_scorer(recall_score, average="micro"),
               "f1": make_scorer(f1_score, average="micro"),
               }
    
    scoresMacro = cross_validate(net, train_sliceable, yTrain, cv=kf, scoring=scoringMacro)
    scoresMicro = cross_validate(net, train_sliceable, yTrain, cv=kf, scoring=scoringMicro)
    
    #prints out all k-fold values
    #Macro scores
    print("Macro scores:")
    for metrics, values in scoresMacro.items():
        print(f"{metrics.capitalize()} scores: \n{values}")

    #Micro scores
    print("\nMicro scores:")
    for metrics, values in scoresMicro.items():
        print(f"{metrics.capitalize()} scores: \n{values}")

   
#To run python script
if __name__ == "__main__":
    #Set random seed to be the same each time to help with reproducability
    torch.manual_seed(0)
    np.random.seed(0)  

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

    #model = CNNV1() #variant 1
    #model = CNNV2() #variant 2
    model = CNN()
    model.load_state_dict(torch.load(loadModelPath))
    # evaluate(model,testData,trainData,classes)
    kFoldCrossValidation(model,trainData,yTrain,k=10)