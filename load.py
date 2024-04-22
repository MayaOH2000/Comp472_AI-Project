import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from cnnModel import CNN
from cnnModel import CNNV1
from cnnModel import CNNV2


#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model"   #main model
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv1"   #V1
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv2"   #V2

#trying to mitigate bias
loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelBias"    #main no bias?
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelBias_best.pth"

#best fit models
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/model_best.pth"   #main model
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv1_best.pth"   #V1
#loadModelPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Models/modelv2_best.pth"   #V2

#Data sets
#Original
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/train"    #orignal

#bias
#age
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/age-bias-train/middle"    #age/middle
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/age-bias-train/senior"    #age/senior
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/age-bias-train/young"    #age/young

#gender 
#dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/gender/Female"   #gender/Female
dataPath = "C:/Users/mayao/Desktop/Comp 472 - Artificiall intelligence/Project/Comp472_AI-Project/Dataset/gender/Male"     #gender/Male

if __name__ == "__main__":
    #Set random seed to be the same each time to help with reproducability
    torch.manual_seed(0)
    np.random.seed(0) 

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
    # testDataset = torchvision.datasets.ImageFolder(root=dataPath, transform=transform)
    # testLoader = DataLoader(testDataset, batch_size=32)

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

            #Compares predicted and true label and updates counter
            correctPred += (predicted == labels).sum().item()

    # Calculate accuracy
    print("+++++ Dataset Accuracy ++++++")
    print("Accuracy: %.2f" % (100 * (correctPred / totalImg)))


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
