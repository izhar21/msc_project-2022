import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Conv2D, ReLU, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras import backend as K


os.chdir('"C:/Users/izhar/Desktop/Image_dataset.csv')
dim=(256,256)
imageShape = (dim[0],dim[1],3)
numClasses = 2
batchSize = 10
epochs = 2
folderWithPics='Image_dataset.csv'
dirs=os.listdir('./'+folderWithPics)
clsLabels=pd.read_csv('./'+folderWithPics+'/groundTruthLabel.txt',delimiter='\t')
clsLabels.index=clsLabels.index+1
subDirPath=[('./'+folderWithPics+'/'+di) for di in dirs if('txt' not in di)]
allImagesTrainPath=[(si+'/'+ii) for si in subDirPath[:-1] for ii in os.listdir(si) if('jpg' in ii)]
allImagesTestPath=[(si+'/'+ii) for si in [subDirPath[-1]] for ii in os.listdir(si) if('jpg' in ii)]


def formImageSet(allImagesFoldrPath,dim,clsLabels):
    x_imageSet=np.empty((len(allImagesFoldrPath),dim[0],dim[1],3))
    y_Set=np.empty((len(allImagesFoldrPath),1))
    for im in range(len(allImagesFoldrPath)):
        readImage=imread(allImagesFoldrPath[im])
        
        imNum=int(allImagesFoldrPath[im].split('/')[-1].split('.')[0])
        actualClass=clsLabels.loc[imNum][1]
        
        if (actualClass=='positive'):
            y_Set[im]=1
        else:
            y_Set[im]=0
            
        if (len(readImage.shape)>=3):
            if readImage.shape[2]>3:
                readImage=readImage[:,:,:3]            
        else:
            print(im,readImage.shape)
            readImage=gray2rgb(readImage)            
        readImage=resize(readImage,dim)
        x_imageSet[im]=readImage
    return x_imageSet,y_Set

def prepareDataSet():
    xTrainImSet,yTrainSet=formImageSet(allImagesTrainPath,dim,clsLabels)
    xTestImSet,yTestSet=formImageSet(allImagesTestPath,dim,clsLabels)
    
    xTrainImSet= xTrainImSet.astype('float32')
    xTestImSet= xTestImSet.astype('float32')
    xTrainImSet /= 255.0
    xTestImSet /= 255.0
    
# Categorical representation by converting the class vectors to matrices as binary
    yTrainSet= keras.utils.to_categorical(yTrainSet, numClasses)
    yTestSet= keras.utils.to_categorical(yTestSet, numClasses)
    
    print('Train Dataset size: ', xTrainImSet.shape[0])
    print('Test Dataset size: ', yTestSet.shape[0])
    
    return (xTrainImSet,yTrainSet), (xTestImSet,yTestSet)

# Desing DenseNet architecture 
def createAModel():
# The sequential model of keras is used
    model = model()
    
# 1st Convolution layer with 64 filters
    model.add(Conv2D(64, kernel_size=(7,7), padding='same', 
					 kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros', 
                     input_shape=imageShape))
    model.add(Activation('relu'))       
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# 1st Dense block
    def dense_factor(inputs):
        h_1 = BatchNormalization()(inputs)
        h_1 = Conv2D(6, (3,3), border_mode='same')(h_1)
        output = ReLU()(h_1)
        return output

    def dense_block(inputs):

        concatenated_inputs = inputs

        for i in range(3):
             x = dense_factor(concatenated_inputs)
             concatenated_inputs = concatenated_inputs([concatenated_inputs, x], axis=3)

             return concatenated_inputs   

    def dense_block_model(x_train):
        inputs = inputs(x_train.shape[1:])

        x = dense_block(inputs)
        x = Flatten()(x)
        x = Dense(64, activation='softmax')(x)
        predictions = Dense(10, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.SGD(lr=1e-2, decay=1e-6),
                    metrics=['accuracy'])
        return model          
# 2nd Convolution layer with 64 filters
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))   

# 2nd Dense block
    def dense_factor(inputs):
        h_1 = BatchNormalization()(inputs)
        h_1 = Conv2D(12, (3,3), border_mode='same')(h_1)
        output = ReLU()(h_1)
        return output

    def dense_block(inputs):

        concatenated_inputs = inputs

        for i in range(3):
             x = dense_factor(concatenated_inputs)
             concatenated_inputs = concatenated_inputs([concatenated_inputs, x], axis=3)

             return concatenated_inputs   

    def dense_block_model(x_train):
        inputs = inputs(x_train.shape[1:])

        x = dense_block(inputs)
        x = Flatten()(x)
        x = Dense(64, activation='softmax')(x)
        predictions = Dense(10, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.SGD(lr=1e-2, decay=1e-6),
                    metrics=['accuracy'])
        return model               
              
#3rd Convolution layer with 128 filters
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))

# 3rd Dense block
    def dense_factor(inputs):
        h_1 = BatchNormalization()(inputs)
        h_1 = Conv2D(24, (3,3), border_mode='same')(h_1)
        output = ReLU()(h_1)
        return output

    def dense_block(inputs):

        concatenated_inputs = inputs

        for i in range(3):
             x = dense_factor(concatenated_inputs)
             concatenated_inputs = concatenated_inputs([concatenated_inputs, x], axis=3)

             return concatenated_inputs   

    def dense_block_model(x_train):
        inputs = inputs(x_train.shape[1:])

        x = dense_block(inputs)
        x = Flatten()(x)
        x = Dense(64, activation='softmax')(x)
        predictions = Dense(10, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.SGD(lr=1e-2, decay=1e-6),
                    metrics=['accuracy'])
        return model               
                   
#4th Convolution layer with 128 filters
    model.add(Conv2D(128, kernel_size=(3,3), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# 4th Dense block
    def dense_factor(inputs):
        h_1 = BatchNormalization()(inputs)
        h_1 = Conv2D(16, (3,3), border_mode='same')(h_1)
        output = ReLU()(h_1)
        return output

    def dense_block(inputs):

        concatenated_inputs = inputs

        for i in range(3):
             x = dense_factor(concatenated_inputs)
             concatenated_inputs = concatenated_inputs([concatenated_inputs, x], axis=3)

             return concatenated_inputs   

    def dense_block_model(x_train):
        inputs = inputs(x_train.shape[1:])

        x = dense_block(inputs)
        x = Flatten()(x)
        x = Dense(64, activation='softmax')(x)
        predictions = Dense(10, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.SGD(lr=1e-2, decay=1e-6),
                    metrics=['accuracy'])
        return model                   
    
#5th Convolution layer with 512 filters
    model.add(Conv2D(512, kernel_size=(1,1), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))                      
    model.add(AveragePooling2D(pool_size=(7,7)))

# Flatten layer
    model.add(Flatten())
    
# Dense layer 
    model.add(Dense(numClasses, kernel_initializer='glorot_uniform', bias_initializer='1000'))
    model.add(Activation('softmax'))
    
    return model

# prepare the data set
print('Prepare data set...')
(xTrainImSet,yTrainSet), (xTestImSet,yTestSet) = prepareDataSet()
    
# Create a model
print('Create a model...')
model = createAModel()

# Set the optimizer and compile the model
print('Set the optimizer and compile the model')
optimizer = SGD(0.01, 0.8, 0.0005, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
print('Train the model')
model.fit(xTrainImSet, yTrainSet,batch_size=batchSize,epochs=epochs,validation_data=(xTestImSet, yTestSet),shuffle=True)

print('Tesing the model')
score = model.evaluate(xTestImSet, yTestSet)
print('Test accuracy: ', score[1],'Test loss: ', score[0])