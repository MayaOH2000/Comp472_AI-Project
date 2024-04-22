from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, make_scorer, precision_score,recall_score,f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Subset
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
from train import train

#change path to match data path and model to evaluate
loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model"   #main model
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv1"   #V1
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv2"   #V2

#best fit models
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model_best.pth"   #main model
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv1_best.pth"   #V1
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv2_best.pth"   #V2


modelPath1 = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model2"    #main

#Data sets
#Original
dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"    #orignal

#bias
#age
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/age-bias-train/middle"    #age/middle
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/age-bias-train/senior"    #age/senior
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/age-bias-train/young"    #age/young

#gender 
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/gender/Female"   #gender/Female
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/gender/Male"     #gender/Male

"""Evaluation with Confusion Matrix"""
def evaluate(cnnModel,testLoader, classes ):  
    #Evaluation with Test Data
    cnnModel.eval()
    yTest = []
    yPredict = []

    with torch.no_grad():
        for images, labels in testLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            yTest.extend(labels.detach().cpu().numpy())
            yPredict.extend(predicted.detach().cpu().numpy())

    #Printing report out for recall,precision, f1-score and accuracy of model
    print(classification_report(yTest, yPredict))

    #accuracy for test
    print("Accuracy for {} Dataset: {}%" .format("Test" , round(accuracy_score(yTest,yPredict)*100,2)))

    #confusion matrix
    ConfusionMatrixDisplay.from_predictions(yTest, yPredict, display_labels=classes)
    plt.title('Confusion Matrix for {} Dataset'.format('Test'))
    plt.show()
 
"""K-fold Cross-Validate with k = 10 folds"""

def trainKFold(model, trainLoader, valLoader, criterion, optimizer, device, num_epochs, patience = 5):
     #Early stopping variables
    bestValLoss = float('inf')
    patience = 5
    noImprovementCount = 0

    #Best fit variables
    #best-fit save path
    # bestModel = modelPath + "_best.pth"

    #training model
    #training loop
    for epoch in range(num_epochs):
        #training phase
        model.train()  
        trainCorrect = 0
        trainTotal = 0
        trainLoss = 0.0
        for i, (images,labels) in enumerate(trainLoader):
            images, labels = images.to(device), labels.to(device)
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
            # torch.save(model.state_dict(), bestModel)
            noImprovementCount = 0
        else:
            noImprovementCount += 1

        #Early stopping
        if noImprovementCount > patience:
            print("*** Early Stopping Happened! ****")
            break

    # #saving the final model
    # torch.save(model.state_dict(), modelPath)
     
    print("\n ++++++ Training Complete!! +++++ ")

#Takes a long time to compute
def kFoldCrossValidation(cnnModel,trainData,valLoader,k = 10):
    
    #Hyperparameters
    num_epochs = 10     #minimum of 10 iteration

    kf = KFold(n_splits=k, shuffle = True, random_state = 0)

    # #store the macro metrics
    macroAccuracy = 0.0
    macroPrecision = 0.0
    macroRecall = 0.0
    macroF1 = 0.0

    #micro metric
    microAccuracy = 0.0
    microPrecision = 0.0
    microRecall = 0.0
    microF1 = 0.0

    #train 
    cnnModel.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    #k-fold loop
    for fold, (trainIndex, _) in enumerate(kf.split(trainData),1): #start it at 1 
        print("===========================================")
        print(f"Fold {fold}/ {k}:")
        print("========================================")
        trainSet = Subset(trainData, trainIndex) #creates subset of train data for eack k fold
        trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True, num_workers=2)
        
        #train the model
        trainKFold(cnnModel, trainLoader,valLoader,criterion,optimizer,device, num_epochs)

        #evalute 
        cnnModel.eval()
        valLoss = 0.0
        valCorrect = 0
        valTotal = 0
        yTrue = []
        yPredict = []

        with torch.no_grad():
            for images, labels in valLoader:
                images,labels = images.to(device), labels.to(device)
                outputs = cnnModel(images)
                loss = criterion(outputs, labels)
                
                #val loss
                valLoss += loss.item() * images.size(0)

                #val accuracy
                _, predicted = torch.max(outputs.data, 1)
                valTotal += labels.size(0)
                valCorrect += (predicted == labels).sum().item()
                yTrue.extend(labels.cpu().numpy())
                yPredict.extend(predicted.cpu().numpy())
            
            #Metrics Calculations Macro
            accuracy = accuracy_score(yTrue, yPredict)
            precision = precision_score (yTrue, yPredict, average='macro')
            recall = recall_score (yTrue, yPredict, average='macro')
            f1 = f1_score (yTrue, yPredict, average='macro')

            #printing k-fold results 
            #Macro
            print("Macro Values: ")
            print(f"Validation Loss {valLoss / len (valLoader.dataset):.4f}, Accuracy: {accuracy *100:.2f}%, "
                  f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
            
            #Metrics Calculations micro
            accuracyMic = accuracy_score(yTrue, yPredict)
            precisionMic = precision_score (yTrue, yPredict, average='micro')
            recallMic = recall_score (yTrue, yPredict, average='micro')
            f1Mic = f1_score (yTrue, yPredict, average='micro')

            #printing k-fold results 
            #Macro
            print("\nMicro Values: ")
            print(f"Validation Loss {valLoss / len (valLoader.dataset):.4f}, Accuracy: {accuracyMic *100:.2f}%, "
                  f"Precision: {precisionMic:.2f}, Recall: {recallMic:.2f}, F1-Score: {f1Mic:.2f}")


            #Total fold metrics 
            macroAccuracy += accuracy
            macroPrecision += precision
            macroRecall += recall
            macroF1 += f1

            microAccuracy += accuracyMic
            microPrecision += precisionMic
            microRecall += recallMic
            microF1 += f1Mic

    #average values of total folds
    averageMacroAccuracy = macroAccuracy/k
    averageMacroPrecision = macroPrecision/k
    averageMacroRecall = macroRecall/k
    averageMacroF1 = macroF1/k

    averageMicroAccuracy = microAccuracy/k
    averageMicroPrecision = microPrecision/k
    averageMicroRecall = microRecall/k
    averageMicroF1 = microF1/k

    #print K-fold cross validation
    print(f"\nK-Fold Cross Validation Average Results for {k} Folds")
    print("===============================================")
    print("Macro Averages: ")
    print(f"Accuracy: {averageMacroAccuracy *100:.2f}%, Precision: {averageMacroPrecision:.2f}, "
          f"Recall: {averageMacroRecall:.2f}, F1-Score: {averageMacroF1:.2f}")

    print("\nMicro Average Values: ")
    print(f"Accuracy: {averageMicroAccuracy *100:.2f}%, Precision: {averageMicroPrecision:.2f}, "
          f"Recall: {averageMicroRecall:.2f}, F1-Score: {averageMicroF1:.2f}")

   
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

    #getting dataset for model to be evaluated on
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

    #model = CNNV1() #variant 1
    #model = CNNV2() #variant 2
    model = CNN()
    model.load_state_dict(torch.load(loadModelPath))

    evaluate(model,testLoader,classes)
    #kFoldCrossValidation(model,trainData,valLoader, k=10)