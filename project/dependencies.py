"""Import dependencies for the notebook"""

import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import cv2

from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
