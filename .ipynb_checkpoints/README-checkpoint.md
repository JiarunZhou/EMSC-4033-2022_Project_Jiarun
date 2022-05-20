# Data-Preprocessing Module

This repository is for an assessment of EMSC-4033-2022, in which I build a cloud image data preprocessing module for deep-learning models. In this project, cloud image dataset is imported and goes through some data processing procedures and data augmentation. Finally, this module outputs a training image genetator and a validating image generator, which can be input into a deep-learning model for use directly.

In this repository, I use a cloud image dataset containing 1200 images collected from the Kerguelen Plateau. If the users would like to use other datasets with different formats, please check the function settings.

In data **processing**, this module achieves:

 - Resizing and cropping images
 - shuffling datasets
 - Converting labels to one hot matrices
 - Split of training, test and validation sets
 - Subtracting mean RGB values on the training set
 
In data **augmentation**, this module can perform the following operations on images:

 - ZCA_whitening
 - Rotation 
 - Width and height shift
 - Horizontal and vertical flip
 
## Some imported image samples

!["Can't find the plot"](random_images.png)

## **Welcome to leaving your ideas, suggestions and questions.**




