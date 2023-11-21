import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling3D, UpSampling3D, Activation, Dense, Multiply, Reshape, Flatten,Concatenate
from keras.layers import Conv2DTranspose, ConvLSTM2D, Lambda, Average, Bidirectional, Add, Dropout, GlobalMaxPool1D, GlobalAveragePooling2D, GlobalAveragePooling1D
from keras_layer_normalization import LayerNormalization
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

                        
def split_bag(x):
    
    c0, c1, c2, c3 = tf.split(x, [1,1,1,1], 1)
    
        
    return c0, c1, c2, c3

def split_prob(x):
    
    c0, c1, c2 = tf.split(x, [1,1,1], 1)
    
        
    return c0, c1, c2

def check_score(x):
    
    sc_index = tf.math.argmax(x)
    
    return sc_index
    

def build_convnet(shape=(128, 128, 3)):
    input_tensor = Input(shape, name='input')
    base_model = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=Input(shape=(128, 128, 3)),
    input_shape=None,
    pooling=None,
    classes=3)
    
    headModel = base_model.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(4096, activation="relu")(headModel)
    headModel = Dense(4096, activation="relu")(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    return tf.keras.models.Model(inputs=base_model.input, outputs=headModel)
                        

def build_MIL_convnet(shape=(4, 128, 128, 3)):

    headmodel = build_convnet(shape=(128, 128, 3))
    headmodel.summary()
    bag_size = 4 


    file_name_model = "VGG_clinical.hdf5"
    headmodel.load_weights(file_name_model)

                                               
    headmodel = Model(inputs=headmodel.input, outputs=headmodel.layers[len(headmodel.layers)-2].output)


    input_img = Input(shape = (4, 128, 128, 3))  #input data
    TD = tf.keras.layers.TimeDistributed(headmodel)(input_img)
    Bag_fine = tf.keras.layers.TimeDistributed(Dense(256, activation = "relu"))(TD)
    Bag_fine_label = tf.keras.layers.TimeDistributed(Dense(3, activation = "softmax"), name = 'Fine_label')(Bag_fine)


    B1, B2, B3, B4 = Lambda(split_bag, name ='split_bag')(Bag_fine_label) 

    B1 = Flatten(name="B1")(B1)
    P1_B1, P2_B1, P3_B1 = Lambda(split_prob, name ='split_prob1')(B1)
    B2 = Flatten(name="B2")(B2)
    P1_B2, P2_B2, P3_B2 = Lambda(split_prob, name ='split_prob2')(B2)
    B3 = Flatten(name="B3")(B3)
    P1_B3, P2_B3, P3_B3 = Lambda(split_prob, name ='split_prob3')(B3)
    B4 = Flatten(name="B4")(B4)
    P1_B4, P2_B4, P3_B4 = Lambda(split_prob, name ='split_prob4')(B4)

    P_cancerous_B1 = Add()([P1_B1, P2_B1])
    P_cancerous_B2 = Add()([P1_B2, P2_B2])
    P_cancerous_B3 = Add()([P1_B3, P2_B3])
    P_cancerous_B4 = Add()([P1_B4, P2_B4])

    P_cancerous = Average()([P_cancerous_B1, P_cancerous_B2, P_cancerous_B3, P_cancerous_B4])
    P_nonCancerous = Average()([P3_B1, P3_B2, P3_B3, P3_B4])
    Bag_score = Concatenate(axis=-1, name = 'Bag_label')([P_cancerous, P_nonCancerous])


    model = Model(inputs=input_img, outputs=[Bag_score, Bag_fine_label])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00000001, beta_1=0.9, beta_2=0.999, amsgrad=False), loss=["CategoricalCrossentropy", "CategoricalCrossentropy"])

    model.summary()
    return model
