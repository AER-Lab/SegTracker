#This file compiles and saves a Unet model 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class UnetModel():
    def __init__(self):    
        #input layer
        model_input = keras.layers.Input((256,256,3))
        # 3 downsamples, each envolving a doubling in 
        c0, mp0 = self.downSample(model_input, 8) 
        c1, mp1 = self.downSample(mp0, 16) 
        c2, mp2 = self.downSample(mp1, 32) 
        c3, mp3 = self.downSample(mp2, 64)
        BN = self.bottleneck(mp3, 128)
        u1 =  self.upSample(BN, c3, 64)
        u2 =  self.upSample(u1, c2, 32)
        u3 =  self.upSample(u2, c1, 16)
        u3 =  self.upSample(u3, c0, 8)

        model_output = keras.layers.Conv2D(3, (1, 1), padding="same", activation="sigmoid")(u3)
        self.model = keras.models.Model(model_input, model_output)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        
    def downSample(self,input_image, num_filters):
        convolution = keras.layers.Conv2D(num_filters,(3,3),activation = 'relu', padding = 'same')(input_image)
        convolution = keras.layers.Dropout(.1)(convolution)
        #convolution = keras.layers.Conv2D(num_filters,(3,3),activation = 'relu', padding = 'same')(convolution)
        max_pool = keras.layers.MaxPool2D((2, 2), (2, 2))(convolution)
        return convolution, max_pool

    def upSample(self,input_image, concat, num_filters):
        upsampled_image = keras.layers.UpSampling2D((2, 2))(input_image)
        concatinated_image = keras.layers.Concatenate()([upsampled_image, concat])
        output_image = keras.layers.Conv2D(num_filters,(3,3), padding='same', activation="relu")(concatinated_image)
        output_image = keras.layers.Conv2D(num_filters,(3,3), padding='same', activation="relu")(output_image)
        return output_image

    def bottleneck(self,input_image, num_filters):
        bottleNeck_image = keras.layers.Conv2D(num_filters,(3,3) , padding='same', activation="relu")(input_image)
        bottleNeck_image = keras.layers.Conv2D(num_filters,(3,3), padding='same', activation="relu")(bottleNeck_image)
        return bottleNeck_image
    
    def getModel(self):
        return self.model
        
