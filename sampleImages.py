import os
import random
import matplotlib.pyplot as plt
from PIL import Image

#Resizing image here and converting it to grayscale to match actual dataset representation
def imageResize(imagePath, size = (75, 75)):
    #L is used to conver images to grayscale
    image = Image.open(imagePath).convert('L')
    image = image.resize(size)
    return image

#plot image grid in a 5 x 5 grid to fit on letter page (8.5 x 11)
def plotImageGrid(images, title):
    fig, axes = plt.subplots(5,5,figsize=(8.5,11))
    fig.suptitle(title, fontsize = 24)

    # 5 x 5 grid
    for i, _ in enumerate(axes.flat):
        _.axis('off')
        if i < len(images):
            imgPath = images[i]
            image = imageResize(imgPath)
            _.imshow(image, cmap='gray')
            # #plot histogram graph under image
            # pixale = list(image.getdata())
            # hisAxis = fig.add_axes([_.get_position().x0, _.get_position().y0 - 0.1, _.get_position().width,0.1])
            # hisAxis.hist(pixale, 256 , [0,256], True)
            # hisAxis.axis('off')

    plt.tight_layout(rect=[0,0,1,1]) #padding layout of plot grid   
    # plt.hist(list(image.getdata()),256,[0,256], True)
    plt.show()


def plotHistogram(images, title):
    fig, axes = plt.subplots(5,5,figsize=(8.5,11))
    fig.suptitle(title, fontsize = 24)

    # 5 x 5 grid
    for i, _ in enumerate(axes.flat):
        _.axis('off')
        if i < len(images):
            imgPath = images[i]
            image = imageResize(imgPath)
            #plot histogram 
            pixale = list(image.getdata())
            #adding axis location to plot
            hisAxis = fig.add_axes([_.get_position().x0, _.get_position().y0, _.get_position().width,0.1])  
            hisAxis.hist(pixale, 256 , [0,256], True)
            
    plt.tight_layout(rect=[0,0,0.9,1]) #padding layout of plot grid   
    plt.show()


def getRandomImages(classPath,imageNumber = 25):
    imageFile = [file for file in os.listdir(classPath) 
                 if file.endswith ('.jpg')]
    randomImages = random.sample(imageFile, min(imageNumber, len(imageFile)))
    return [os.path.join(classPath,image) for image in randomImages]

def displayRandomImages(datasetPath):
    for className in os.listdir(datasetPath):
        classPath = os.path.join(datasetPath, className)
        if os.path.isdir(classPath):
            randomImages = getRandomImages(classPath)
            plotImageGrid(randomImages, f"Randome Images from {className}")
            #Display histogram
            plotHistogram(randomImages, f"Histogram from 5 x 5 grid from {className}")

if __name__ == "__main__":
    datasetPath = "Dataset\\train"
    #Display grid for each class
    displayRandomImages(datasetPath)
