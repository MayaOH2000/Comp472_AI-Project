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
def confusionMatrix(model, trainData, testData )
#     #model fitting
#     y_train = np.array([np.int64(y) for x, y in iter(trainData)])
#     model.fit(trainData, y=y_train)
#     print("\nFinished Training!!")
            
#     #testing model that just been trained
#     y_predict = model.predict(testData)
#     y_test = np.array([y for x, y in iter(testData)])

#     accuracy = accuracy_score(y_test, y_predict)
#     confusion_matrix(model, testData, y_test.reshape(-1, 1))
#     plt.show()


# # print('Accuracy for {} Dataset: {}%'.format('Test', round(accuracy_score(y_test, y_predict) * 100, 2)))
# # ConfusionMatrixDisplay.from_predictions(y_test, y_predict, display_labels=classes)
# # plt.title('Confusion Matrix for {} Dataset'.format('Test'))
# # plt.show()

#     precision = precision_score(y_test, y_predict)

#     print(classification_report())
# 