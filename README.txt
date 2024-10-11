CNN classification for Fashion MNIST

This project builds a Convolutional Neural Network (CNN) with Keras through Python to recognize the Fashion MNIST dataset. This dataset has four types of fashion attributes including shirts, shoes, bags, and trousers. The correct category for each of them is learned and predictions are made on test images used in the model.


## Dataset
Fashion MNIST has 60,000 training images and 10,000 test images of dimension 28 x 28 pixels in greyscale. Each image belongs to one of the following 10 categories:
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot

## Project Overview
The CNN architecture used consists of six layers:

Convolutional layers to feature extraction from the image.
We use MaxPooling layers to decrease the size of the feature maps in order to simplify the subsequent computational processing.

A layer of connected network to make the right category on the images.
First based on the training data set and then based on the test data set the model is trained and tested. The last procedure is to make predictions on at least two images and check their true labels of prediction.

## Instructions
Make sure you have the following libraries installed:
TensorFlow (version 2.17.0 or higher)
Keras (comes with TensorFlow)
Matplotlib for visualizations
NumPy or the fundamental package for high-performance and optimized computing for numerical operations
You can install these using the following commands:

## Usage

install tensorflow using pip install tensorflow, matplotlib using pip install matplotlib and numpy using pip install numpy.
Running the Code
It is advised to copy the repository or download the project files into a directory of your computer.
Navigate to the project directory and run.


## Example Results

Model Accuracy: The model performs approximate to X% ACC on the test set (again update it with actual value once you run the code).

Confusion Matrix: In order to evaluate how accurate the performed model is in discriminating the various categories of fashion products, a confusion matrix is created.

Predictions: The predictions of the model are described on two images from the test set. For example:

Image 1: True = Sneaker, Pred = Sneaker
Image 2: True label = True label = ‘T-shirt/top’, Predicted label = ‘T-shirt/top’


## Interpretation of Results

CNN model also is shown to classify most of the images from Fashion MNIST dataset accurately. However, like with virtually all the machine learning models when they are first rolled out, it is not infallible. There will occasionally be misclassifications arising out of the similarity of some classes, for instance shirt and pullover. Conflict matrix aids in establishing which classes are probably to be more confusable.

## Conclusion
This project best illustrates how effectiveness of Convolutional Neural Networks in a given image classification problem. As we can see, with a simple CNN structure, we can obtain a higher accuracy rate for Fashion MNIST data set.