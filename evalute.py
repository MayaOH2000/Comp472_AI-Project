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

loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model"
dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"

def evaluate(cnnModel, testData, classes ):
    #Evaluating model that just been trained
    y_predict = cnnModel.predict(testData)
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
    
    evaluate()