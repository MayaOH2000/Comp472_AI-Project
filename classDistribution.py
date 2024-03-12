import os
import matplotlib.pyplot as plt

#Count number of images in each class
def imagesInClass(directory):
    classCounts = {}

    for className in os.listdir(directory):
        classPath = os.path.join(directory, className)
        if os.path.isdir(classPath):
            imageCount = len([file for file in os.listdir(classPath) 
                             if file.endswith('.jpg')])
            classCounts[className] = imageCount
    return classCounts
    
#Bar graph to plot 
def plotBarGraph(classCounts):
    classes = list(classCounts.keys())
    counts = list(classCounts.values())
    #variables and parameter of bar graph
    plt.bar(classes,counts, color='green')
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.title("Total Amount of Images in Each Class")
    plt.show()

#To run python script
if __name__ == "__main__":
    #file path directory for all labelled classes
    imageDirectory = 'Dataset\\train'
    #Display total images per class in terminal
    classCounts = imagesInClass(imageDirectory)
    print("Total images in classes:")
    for className, count in classCounts.items():
        print(f"{className}:  {count} images")
    #Display bar graph 
    plotBarGraph(classCounts)
