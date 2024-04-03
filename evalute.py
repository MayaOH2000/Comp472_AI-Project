from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
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

#change path to match data path and model to evaluate
loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model"   #main model
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv1"   #V1
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv2"   #V2

dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from skorch import NeuralNetClassifier
import torch


def evaluate(cnnModel, testData,trainData, classes ):  
    torch.manual_seed(0)
    net = NeuralNetClassifier(
        cnnModel,
        max_epochs=10,
        lr=0.001,
        batch_size=32,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
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
    evaluate(model,testData,trainData,classes)