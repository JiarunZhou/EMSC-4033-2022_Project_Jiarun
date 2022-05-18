"""
Define test functions for each of functions:
    - get_images()
    - display_random_image()
    - subtract_meanRGB()
    - Image_Generator()
"""

import pytest
from functions import *
from dependencies import *

def test_get_images(type_c = np.ndarray): 
    
    Images, Labels, classes = get_images('src/K-Axis Cloud Types', random_value = 1)
    
    # Test with the number of elements
    n_image = Images.shape[0]
    n_label = Labels.shape[0]
    
    # Test with the type of *classes*
    type_class = type(classes)
    
    assert n_image == n_label and type_class == type_c, " *** Fail to return correct images, labels and classes. "
    
def test_display_random_image(Type_fig = matplotlib.figure.Figure):
    
    Images, Labels, classes = get_images('src/K-Axis Cloud Types', random_value = 1)
    
    Images_1 = Images/255.
    
    Fig = display_random_image(classes,Images_1,Labels)
    
    # Test with the figure type
    type_fig = type(Fig)
    
    assert type_fig == Type_fig, " *** Fail to return the figure. "
    
def test_subtract_meanRGB(mean_RGB = 1.e-10):
    # If meanRGB values are substracted, mean values of the return array shoule be 0
    
    # Return a random array for use
    a_random = np.random.random((100,227,227,3))
    
    a_mean = subtract_meanRGB(a_random)
    
    imagenumber = a_mean.shape[0]
    imagesize = a_mean.shape[1]
    
    # Calculate mean value of R channel
    mean_R_1 = np.zeros(imagenumber)
    
    # Extract mean values of each element in this random array
    for i in range(imagenumber):
        mean_R_1[i] = np.mean(a_mean[i,:,:,0])
    
    # Overall mean 
    mean_R = np.mean(mean_R_1)
    abs_mean_R = abs(mean_R)
    
    assert abs_mean_R < mean_RGB, " *** Fail to return the processed array with meanRGB = 0"
    
def test_Image_Generator(Type = keras.preprocessing.image.NumpyArrayIterator):
    
    # Return random images and labels for use
    x_random = np.random.random((100,227,227,3))
    y_random = np.random.random((100,4))
    
    Gen = Image_Generator(x_random,y_random,
                               Horizontal_flip=True,
                               Vertical_flip = True,
                               Rotation_range=180)
    
    # Test with the generator type
    type_Gen = type(Gen)
    
    assert type_Gen == Type, " *** Fail to return an image data generator."
    


