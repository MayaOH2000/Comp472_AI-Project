# Comp472_AI-Project
**TEAM AK_18** <br></br>
**Team Members are:**

Maya Ostiguy Hopp - 40175258

Laura Hang - 40203006

Alina Rudani - 40202482

**Extra Info**

The folder called dataset contains all the data that will be used for this project. The train folder contains all the training data and the test folder contains all the test data. Each folder label represents a different class. In total, we have 4 different classes which are neutral, surprise, focused/engaged and angry.

The folder called Models contains all the models that have been trainined using the train.py file. The model with the betd.pth at the end represent the besfit model from training.

Before running any files make sure you have a python ide installed somewhere and have the following libraries installed by using pip install or running the requirements.txt file with pip install -r requirements.txt
files:
- skorch
- matplotlib
- scikit-learn
- numpy
- torch
- torchvision

**Dataset Link**

The dataset used for most classes except for engaged/focused was FER-2013: <a href= "https://www.kaggle.com/datasets/msambare/fer2013">Dataset link here</a>

<br>**Instruction to run code:**

**requirements.txt**

This file contains all the necessary libraries to instsll on your computer to run all the files and code properly. To install this file run pip install -r requirements.txt in the terminal

# Data Visulization files

**classDistribution.py**

The class distribution will show a bar graph of the number of images in each class and will display the number of image count in the Python terminal. If the data path is set to Dataser/train it will do the training on the train dataset.

**sampleimage.py**

The sampleimage.py will provide you with a 5x5 grid with random images from each class and the histogram for each of the images. The set will be displayed 1 by 1. The first will be the 5x5 image grid then when you close the grid the histogram graphs for that set will be displayed. This will continue until all the label datasets been displayed. Only 1 graph set will be displayed until you close the window. Ensure the data path is set to Dataset/train and then you just need to run the Python code for it to work.

# Model files

**cnnModel.py**

The cnnModel.py file contains all the different classes of the cnn models that were used. CNN is the main model, CNNV1 is variant 1 and CNNV2 is variant 2.

**train.py**

The train.py file is were you can train the modle and can see the results of the train model on the test data. You just need to change the datapath and model path to the locations of where you want the dataset to be trained with and where you want the model to be saved. In addition you can comment or un comment one the 3 CNN architecture to train a model with. You can even motify some of the hyperparameters as well. Once chosen the datapath, modelPAth and the CNN architecture, just run the python file. The results should show the training process of the model, the confusion matrix and report result of the train model on the trainData. It will also save the model and besfit model at the end of running the file.

**evaluate.py**

This file allows you to evaluate a model using a certain cnn architecture similar to the train.py file. Just need to comment or uncomment the cnn architecture you want to use and which model you want to evaluate on. Allow you to load models that you wish to evalute as well by just changing the loadModelPath to the path you want to evalute your model with. Also can change the dataset comment or uncomment the datasrt you wich to evluate the model on. The file contains the evaluate function to evaluate and display the confusion matrix of the model chosen along. In addition, the k-fold cross validation function is availbale to run and see the result for each k-fold of the moddle. Just need to comment or uncomment out the evalution and/or kFoldCrossValidation function call based on which one you want to run. Just run the python file and wait for the results.

**load.py**

This file allows you to see the accuracy of a model on a certain dataset and the classification of single image base on the directory given. You just need to change the imagePath and dataPath to correspond to the locations that you want to evaluate the model prediciton on. After that just run the python file.


Git Hub Repo link: https://github.com/MayaOH2000/Comp472_AI-Project.git 


