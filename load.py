import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets
from PIL import Image
from torch.utils.data import DataLoader

from cnnModel import CNN
from cnnModel import CNNV1
from cnnModel import CNNV2

loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model"
dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"

#restore / load model base on save path
#model = CNNV1() #variant 1
#model = CNNV2() #variant 2
model = CNN()    
model.load_state_dict(torch.load(loadModelPath))
model.eval()

#Applying model on dataset
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),   #Images are in grayscaled
    transforms.Resize((48,48)),#image size 48 x 48
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Define dataset and dataloaders
testDataset = torchvision.datasets.ImageFolder(root=dataPath, transform=transform)
testLoader = DataLoader(testDataset, batch_size=32)

# Make predictions on the test dataset
predictions = []
trueLabel = []
correctPred = 0
totalImg = 0
for images, labels in testLoader:
    with torch.no_grad():  
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        totalImg += labels.size(0)

        #Compres predicted and true label and updates counter
        correctPred += (predicted == labels).sum().item()

# Calculate accuracy
print("+++++ Dataset Accuracy ++++++")
print("Accuracy: %d %%" % (100 * correctPred / totalImg))


"""Begins the single image prediction"""
#Classes to classify into 
classes = ("angry", "neutral", "engaged", "surpise")

# Define transforms
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),   #Images are in grayscaled
    transforms.Resize((48,48)),#image size 48 x 48
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

#Image path to be tested with model
imagePath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train/angry/PrivateTest_731447.jpg"
image = Image.open(imagePath)

# Transform image to tensorflow
imageTensor = transform(image).unsqueeze(0)  # Add batch dimension

#Individual image prediction
with torch.no_grad():
    model.eval() 
    output = model(imageTensor)
    _, predicted_class = torch.max(output, 1)

# Map the predicted class index to the corresponding emotion label
predicted_emotion = classes[predicted_class.item()]
print("\n++++++Single Immage prediction+++++++")
print("Predicted Emotion:", predicted_emotion)
