# EMSC4033 project plan template

## Project title

An integrated module of cloud image data preprocessing for deep learning

## Executive summary

Classifying cloud types in the atmosphere particularly over the Southern Ocean is helpful to improve the simulation accuracy of climate models. Deep learning provides a automated method to process a huge amount of cloud image data. However, a lot of data preprocessing procedures should be done in advance to make sure the deep-learning model can study from the cloud images correctly and efficiently. In this project I plan to build a module in order to do image data 1) input, 2) preprocessing and 3) augmentation.

## Goals

- To preprocess input cloud images into appropriate size and shape based on the requirement of deep-learning models;
- To plot some image examples from a specified cloud type or some types under a clear way;
- To implement dataset split to training, testing and validation sets according to a requested proportion;
- To implement data augmentation operations such as random rotation and horizontal/vertical flip by building a data generator;
- To output the processed variables that can be straightforward used to model fitting.

## Background and Innovation  

As a very popular concept in recent years, deep learning has been introduced into a lot of fields of Earth science study, since there is a high requirement for effecient automated data-processing algorithms with more and more data collected. The most key feature of deep-learning models is they are able to learn data features automatically in training process once the model architecture has been configured well. However, this also indicates the importance of high-quality training datasets to the model performance. This project exactly aims to provide an integrated module to preprocess input data and improve the data quality for deep-learning models.
In previous studies, a variety of data preprocessing methods have been raised. {cite:t}`zhang2018cloudnet` build a cloud image database consisting of 11 categories. A new category "contrails" is added as an innovation. They first split all images into training and test sets and validate there is no overlap between them. Then, the mean activity over the training set is substracted for each pixel, which is benifical to preventing overfitting in the training process. In addition, random flip and crop are also applied to enlarge the total amount of training samples. `2` use a database consisting of 7 categories. All iamge samples in their database are resized to the resolution of 224$\times$224, which is one of the most common input resolutions for deep-learning models. `3` rotate and flip their image samples. They indicate that these operations can demonstrate the insensitivity of cloud images to the spatial layout variation. `4` substract mean RGB value computed on the training set and resize the iamges as well. According to previous studies, data preprocessing is necessary and important to classify cloud images using deep-learning models. Although different choices are made in different studies, there is a lot of overlap in these operations, which indicates we can build some modules and functions beforehand to simpify the procedures of data preprocessing.
There are some existing popular packages for building deep-learning models, in which a few data processing functions are also provided. `scikit-learn`[https://scikit-learn.org] is an abundant Python library for machine learning. It contains various algorithms, such as classification, regression and clustering, and uses `Numpy` for high-performance array calculation. `scikit-learn` also provides fucntions to achive data loading, dataset split and supports the output of confusion matrix for better result analysis and presentation. `Tensorflow`[https://www.tensorflow.org] is an open source platform originally developed by Google, provides comprehensive resources allowing users to build and deploy their deep-leanring applications easily. Developers can build the model architecture and add new layers fro their model by simply calling functions from `Tensoeflow`. This is helpful for the generalization of the deep-learning algorithm. It provides some very useful functions for data preprocessing and data augmentation as well. `ImageDataGenerator` intergrates abundant and diversified  methods for data agmentation, including normalization, whitening, shift and flip, etc. This is particularly essential if the number of original data is limited.
As reviewed above, data preprocessing is necessary and important for deep-learning related tasks, and there have been a lot of methods provided by some popular packages. However, some limitations are revealed as well. First, although model architecture and objectives 
vary in different studies, some data-preprocessing procedures like image reshape and dataset split are required in common. These operations are necessary, but would reduce effeciency in the process of repetition. It would be a better idea to achieve fragmentary data-preprocessing operations within one fucntion. Second, existing functions in the packages reviewed above are poeweful, but always not comprehensive enough for data preprocessing. For example, `ImageDataGenerator`from `Tensorflow.keras` foucuses on data augmentation, but requires unified image format as input, which should be done in advance using other fucntions. Separate calling for each data-preprocessing section facilitates personalized and more detailed adjuestment to the data, while an integrated module is able to simplify a lot of repetitive work.
As a result, this project aims to generate an integrated module for preprocessing raw cloud image dataset in order to help with the training of deep-leanring models. This work would be particularly benificial to determining the most subitable data-preprocessing methods since users can attempt different methods in one single function by simply adjusting function parameters. Although we focuses on cloud images here, it is easy to modify this module and apply it to processing the images of other objects and tha data with different dimensions.
1. CloudNet: Ground‐based cloud classification with deep convolutional neural network
2. Ground-Based Cloud Classification Using Task-Based Graph Convolutional Network
3. DeepCloud: Ground-Based Cloud ImageCategorization Using Deep Convolutional Features
4. Deep Convolutional Activations-Based Featuresfor Ground-Based Cloud Classification

## Resources & Timeline


_What do you have at your disposal already that will help the project along. Did you convince somebody else to help you ? Are there already some packages you can build upon. What makes it possible to do this project in the time available. Do you intend to continue this project in the future ?_

(For example:
  - I’ll be using data of X from satellite and then also data from baby blue seals…
  - I’ll step on existing package Y and build extra functionality on top of class W.
  - I’ll use textbook Z that describes algorithms a, b, c
  -
)

## Testing, validation, documentation



**Note:** You need to think about how you will know your code is correct and achieves the goals that are set out above (specific tests that can be implemented automatically using, for example, the `assert` statement in python.)  It can be really helpful if those tests are also part of the documentation so that when you tell people how to do something with the code, the example you give is specifically targetted by some test code.

_Provide some specific tests with values that you can imagine `assert`ing_

