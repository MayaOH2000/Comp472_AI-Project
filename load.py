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

#restore / load model base on save path
model = CNN()    
model.load_state_dict(torch.load(loadModelPath))

#Applying model on dataset