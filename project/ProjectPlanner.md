# EMSC4033 project plan

## Project title

An integrated module of cloud image data preprocessing for deep learning

## Executive summary

Classifying cloud types in the atmosphere particularly over the Southern Ocean is helpful to improve the simulation accuracy of climate models. Deep learning provides a automated method to process a huge amount of cloud image data. However, a lot of data preprocessing procedures should be done in advance to make sure the deep-learning model can study from the cloud images correctly and efficiently. In this project I plan to build a module in order to do image data 1) input, 2) preprocessing and 3) augmentation.

## Goals

- To preprocess input cloud images into appropriate size and shape based on the requirement of deep-learning models.
- To plot some image examples from a specified cloud type or some types under a clear way.
- To implement dataset split to training, testing and validation sets according to a requested proportion.
- To implement data augmentation operations such as random rotation and horizontal/vertical flip by building a data generator.
- To output the processed variables that can be straightforward used to model fitting.

## Background and Innovation  

As a very popular concept in recent years, deep learning has been introduced into a lot of fields of Earth science study, since there is a high requirement for efficient automated data-processing algorithms with more and more data collected. The most key feature of deep-learning models is they are able to learn data features automatically in training process once the model architecture has been configured well. However, this also indicates the importance of high-quality training datasets to the model performance. This project exactly aims to provide an integrated module to preprocess input data and improve the data quality for deep-learning models.  
  
In previous studies, a variety of data preprocessing methods have been raised. Zhang, Liu et al. (2018)<sup>1</sup> build a cloud image database consisting of 11 categories. A new category "contrails" is added as an innovation. They first split all images into training and test sets and validate there is no overlap between them. Then, the mean activity over the training set is subtracted for each pixel, which is beneficial to preventing overfitting in the training process. In addition, random flip and crop are also applied to enlarge the total amount of training samples. Liu, Li et al. (2020)<sup>2</sup> use a database consisting of 7 categories. All image samples in their database are resized to the resolution of 224\*224, which is one of the most common input resolutions for deep-learning models. Ye, Cao et al. (2017)<sup>3</sup> rotate and flip their image samples. They indicate that these operations can demonstrate the insensitivity of cloud images to the spatial layout variation. Shi, Wang et al. (2017)<sup>4</sup> subtract mean RGB value computed on the training set and resize the images as well. According to previous studies, data preprocessing is necessary and important to classify cloud images using deep-learning models. Although different choices are made in different studies, there is a lot of overlap in these operations, which indicates we can build some modules and functions beforehand to simplify the procedures of data preprocessing. 
  
There are some existing popular packages for building deep-learning models, in which a few data processing functions are also provided. [`sklearn`](https://scikit-learn.org) is an abundant Python library for machine learning. It contains various algorithms, such as classification, regression, and clustering, and uses [`numpy`](https://numpy.org) for high-performance array calculation. [`sklearn`](https://scikit-learn.org) also provides functions to achieve data loading, dataset split and supports the output of confusion matrix for better result analysis and presentation. [`tensorflow`](https://www.tensorflow.org) is an open-source platform originally developed by Google, provides comprehensive resources allowing users to build and deploy their deep-learning applications easily. Developers can build the model architecture and add new layers for their model by simply calling functions from [`tensorflow`](https://www.tensorflow.org). This is helpful for the generalization of the deep-learning algorithm. It provides some very useful functions for data preprocessing and data augmentation as well. [`ImageDataGenerator`](https://keras.io/api/preprocessing/image) integrates abundant and diversified methods for data augmentation, including normalization, whitening, shift, and flip, etc. The image data generator generates an iterator for input data, which does the augmentation work during the process of model training. This is particularly essential if the number of original data is limited.

As reviewed above, data preprocessing is necessary and important for deep-learning related tasks, and there have been a lot of methods provided by some popular packages. However, some limitations are revealed as well. First, although model architecture and objectives 
vary in different studies, some data-preprocessing procedures like image reshape and dataset split are required in common. These operations are necessary but would reduce efficiency in the process of repetition. It would be a better idea to achieve fragmentary data-preprocessing operations within one function. Second, existing functions in the packages reviewed above are powerful, but always not comprehensive enough for data preprocessing. For example, [`ImageDataGenerator`](https://keras.io/api/preprocessing/image) from [`tensorflow.keras`](https://keras.io) focuses on data augmentation, but requires unified image format as input, which should be done in advance using other functions. Separate calling for each data-preprocessing section facilitates personalized and more detailed adjustment to the data, while an integrated module is able to simplify a lot of repetitive work.  
  
As a result, this project aims to generate an integrated module for preprocessing raw cloud image dataset in order to help with the training of deep-learning models. This work would be particularly beneficial to determining the most suitable data-preprocessing methods since users can attempt different methods in one single function by simply adjusting function parameters. Although we focus on cloud images here, it is easy to modify this module and apply it to processing the images of other objects and the data with different dimensions.  
  
### Reference list  

1. Zhang, J., et al. (2018). "CloudNet: Ground‐based cloud classification with deep convolutional neural network." Geophysical Research Letters 45(16): 8665-8672.
2. Liu, S., et al. (2020). "Ground‐based cloud classification using task‐based graph convolutional network." Geophysical Research Letters 47(5): e2020GL087338.
3. Ye, L., et al. (2017). "DeepCloud: Ground-based cloud image categorization using deep convolutional features." IEEE Transactions on Geoscience and Remote Sensing 55(10): 5729-5740.
4. Shi, C., et al. (2017). "Deep convolutional activations-based features for ground-based cloud classification." IEEE Geoscience and Remote Sensing Letters 14(6): 816-820.

## Resources & Timeline

I have done an Honours project that uses deep-learning algorithms to classify Southern clouds. This experience makes me familiar with the construction of deep-learning models and image data preprocessing methods, which would be very helpful for me to achieve the goals of this project in the time available.  

Some existing cloud image data and Python packages make this project achievable in three weeks:  

  - A dataset consisting of 1200 cloud images with the resolution of 720\*576 pixels captured from the Kerguelen Plateau are used in this project to test if the built module can process data correctly.
  - Package [`cv2`](https://pypi.org/project/opencv-python) providing some functions like converting data color and size is used to build the functionality that imports and converts raw images into appropriate format in the module.
  - Input data in this project is stored in an array data type from package [`numpy`](https://numpy.org). This package provides convenient operations of array calculation for data processing.
  - Package [`matplotlib.pyplot`](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.html) is used to plot some image samples, which is helpful for users to see if the data is input correctly.
  - The dataset should be randomly split for training, test and validation, respectively. Package [`sklearn`](https://scikit-learn.org) is used to achieve this functionality.
  - Package [`tensorflow`](https://www.tensorflow.org) provides easy-to-use functions to achieve some data processing operations like transforming category vectors and is used to build an image data generator to achieve the functionality of data augmentation.

In this first week, I will focus on the project plan and look for existing packages that are helpful to achieve the functionality of this data processing module. In the next week I aims to integrate them, and build required new functions. Module structure will be redesigned in this process. In the third week, I will test and validate the designed module and solve potential issues to make sure this module is able to achieve all functions from data input to data preprocessing and augmentation.  

Furthermore, outcomes of this project might be applied into my following Masters project, which aims to detect earthquake events and pick seismic wave phases. Since seismograms have totally different characteristics compared to cloud images, it would be a meaningful but challenging task to fit the integrated data-preprocessing module built in this project to them.

## Testing

Testing work for this project will be implemented in each step:

  - This image dataset will be first input. Whether the cloud class array is returned and the number of elements in the *Image* and *Labels* arrays (the number of them should be consistent) will be tested.
  - Some random image samples will be plotted. Whether a figure is returned correctly will be tested.
  - Mean RGB values calculated from the whole dataset are subtracted from each image. Mean RGB values of the returned array will be tested, which should be 0.
  - The original dataset will be augmented by passing through an defined image data generator. The type of the generated iterator for input data will be tested to make sure it is generated correctly.

If all test functions run successfully, we can believe the module has implemented all functions expected and works well. However, the module's actual performance should be validated through applying it to the training of deep-learning models. This would be a part of future work.

