"""Functions writen for the data-preprocessing module:

    - get_images()
    - display_random_image()
    - subtract_meanRGB()
    - Image_Generator()

"""

from dependencies import *

# Import data
def get_images(directory, IMAGE_SIZE = (284, 227),Crop_Size = 227, random_value = 0):
    
    '''
    Return cloud categories of the dataset and images and labels after resize, cropping, and shuffling.
    IMAGE_SIZE should be reset if the shape of input images has changed.
    Crop_Size should be set according to the requirement of models.
    This functions should be modified if input dataset has different structure of subfolders.
    '''
    
    Images = []
    Labels = []
    classes = []
    i = 0
    # Prepare some values for iamge cropping
    length = int(IMAGE_SIZE[0]/2)
    width = int(IMAGE_SIZE[1]/2)
    Crop_Size_2 = int(Crop_Size/2)
    
    width_l = width-Crop_Size_2
    width_u = width_l + Crop_Size
    
    length_l = length-Crop_Size_2
    length_u = length_l + Crop_Size
    
    # Import images and labels from provided directory
    print("Loading {}".format(directory))
    for folder1 in os.listdir(directory):
        print("-----------------------------------------------------------")
        print(folder1)
        
        for folder2 in os.listdir(os.path.join(directory, folder1)): #Main Directory where each class label is present as folder name.
            
            label = i % 4
            i += 1
            classes.append(folder2)
            
            for file in os.listdir(os.path.join(os.path.join(directory, folder1), folder2)): # Extracting the file name of the image from Class Label folder
                
                image_path = os.path.join(os.path.join(os.path.join(directory, folder1), folder2), file) # Get the path name of the image
                image = cv2.imread(image_path) #Reading the image (OpenCV)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,IMAGE_SIZE) #Resize the image
                image = image[width_l:width_u,length_l:length_u,:] #Crop the image
                
                Images.append(image)
                Labels.append(label)

    Images = np.array(Images, dtype = 'float32')
    Labels = np.array(Labels, dtype = 'int32')
    classes = np.array(classes)
    
    # Shuffle images and labels
    Images_s, Labels_s = shuffle(Images, Labels, random_state = random_value)
    
    return Images_s, Labels_s, np.unique(classes)

# Display some image samples
def display_random_image(class_names, images, labels):
    
    '''
    Return a figure of some image samples.
    Input images should be normalized in advance.
    '''
    
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Some image samples", fontsize=16)
    
    for i in range(25):
        index = np.random.randint(images.shape[0])
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[index])
        plt.title('{} :'.format(index) +class_names[labels[index]])
    
    plt.savefig("random_images")
    plt.show()
    
# Subtract mean RGB values calculated from the training set for each cloud image
def subtract_meanRGB(xtrain):
    
    '''
    Return the images whose mean RGB values have been subtracted from each pixel.
    Input images should have three channels: R, G and B.
    '''

    imagenumber = xtrain.shape[0]
    imagesize = xtrain.shape[1]
    
    mean_R_1 = np.zeros(imagenumber)
    mean_G_1 = np.zeros(imagenumber)
    mean_B_1 = np.zeros(imagenumber)
    X_train_mean = np.zeros((imagenumber,imagesize,imagesize,3))
    
    # Extract mean values of each image
    for i in range(imagenumber):
        mean_R_1[i] = np.mean(xtrain[i,:,:,0])
        mean_G_1[i] = np.mean(xtrain[i,:,:,1])
        mean_B_1[i] = np.mean(xtrain[i,:,:,2])
    
    #Overall mean
    mean_R = np.mean(mean_R_1)
    mean_G = np.mean(mean_G_1)
    mean_B = np.mean(mean_B_1)
    
    print('Mean RGB values:',mean_R,mean_G,mean_B)
    
    #Subtract
    X_train_mean[:,:,:,0] = xtrain[:,:,:,0]-mean_R
    X_train_mean[:,:,:,1] = xtrain[:,:,:,1]-mean_G
    X_train_mean[:,:,:,2] = xtrain[:,:,:,2]-mean_B
    
    return (X_train_mean)

# Define image data generator
def Image_Generator(images,
                    labels,
                    Batch_size=8,
                    Featurewise_center=False,
                    Zca_whitening=False,
                    Rotation_range=0,
                    Width_shift_range=0.0,
                    Height_shift_range=0.0,
                    Horizontal_flip=False,
                    Vertical_flip=False):
    
    '''
    Return a define data generator that can be directly used for training deep-learning models.
    Only a subset of augmentation methods is provided for cloud iamges in this fucntion.
    '''
    
    #Define the generator
    #Only a part of methods is useful for cloud images here.
    datagen = ImageDataGenerator(
        featurewise_center=Featurewise_center,
        zca_whitening=Zca_whitening,
        rotation_range=Rotation_range,
        width_shift_range=Width_shift_range,
        height_shift_range=Height_shift_range,
        horizontal_flip=Horizontal_flip,
        vertical_flip=Vertical_flip
    )
    
    if Zca_whitening == True:
        datagen.fit(images)
    
    # Create the generator through "flow" method
    Generator = datagen.flow(images,labels,batch_size = Batch_size)
    
    return Generator