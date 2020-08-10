import keras.backend as K
from keras.layers import Input, ReLU, Conv1D, GlobalMaxPooling1D, AveragePooling1D,MaxPooling1D, Flatten, TimeDistributed, Dense, \
        BatchNormalization, Lambda,Softmax,Multiply,GlobalAveragePooling1D,Activation,ZeroPadding1D,Add,Concatenate, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform 
from keras.losses import binary_crossentropy,kullback_leibler_divergence
from keras.metrics import binary_accuracy
from keras.utils import to_categorical
from keras.optimizers import Adam

import tensorflow as tf
import numpy as np

def copy_model(model_origin,model_target):
    for l_orig,l_targ in zip(model_origin.layers,model_target.layers):
        l_targ.set_weights(l_orig.get_weights())
    return model_target

def Inception1D(input_shape=(256,1)):
    X_input = Input(input_shape)
    X_input_bn = BatchNormalization(axis=-1)(X_input)

    X = X_input_bn
    X = Conv1D(16,7,strides=2,activation='relu')(X)
    X = MaxPooling1D(3,strides=2)(X)
    X = Conv1D(16,1,strides=1,activation='relu')(X)
    X = Conv1D(16,3,strides=1,activation='relu')(X)
    X = MaxPooling1D(3,strides=2)(X)
    tower_1 = Conv1D(16, 1, padding='same', activation='relu')(X)
    tower_1 = Conv1D(16, 3, padding='same', activation='relu')(tower_1)
    tower_2 = Conv1D(16, 1, padding='same', activation='relu')(X)
    tower_2 = Conv1D(16, 5, padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling1D(3, strides=1, padding='same')(X)
    tower_3 = Conv1D(16, 1, padding='same', activation='relu')(tower_3)
    
    X2 = Concatenate(axis = -1)([tower_1, tower_2, tower_3])
    X2 = MaxPooling1D(3,strides=2)(X2)
    tower_12 = Conv1D(16, 1, padding='same', activation='relu')(X2)
    tower_12 = Conv1D(16, 3, padding='same', activation='relu')(tower_12)
    tower_22 = Conv1D(16, 1, padding='same', activation='relu')(X2)
    tower_22 = Conv1D(16, 5, padding='same', activation='relu')(tower_22)
    tower_32 = MaxPooling1D(3, strides=1, padding='same')(X2)
    tower_32 = Conv1D(16, 1, padding='same', activation='relu')(tower_32)
    X3 = Concatenate(axis = -1)([tower_12, tower_22, tower_32])


    
    global_avp = GlobalAveragePooling1D()(X3)

    model = Model(inputs=[X_input],outputs=[global_avp])
    return model


def model_DL2(trainX, trainy, wd=0.005, lr=0.01, lr_decay=1e-4):
    n_channels, n_timesteps, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    data_input = Input(shape=(n_channels, n_timesteps), dtype='float32', name='input')

    perchan_inp_dim = n_timesteps

    perchan_model = Inception1D((n_timesteps, 1))
    
    # channel-distributed feature extractor block
    data_input_rs = Lambda(lambda x: K.expand_dims(x, axis=-1), name='data_input_rs')(data_input)
    data_processed = TimeDistributed(perchan_model, name='data_before_mil')(data_input_rs)
    #attention block
    data_attention = TimeDistributed(Dense(32, activation='tanh', kernel_regularizer=l2(wd), use_bias=False))(
        data_processed)
    data_attention = TimeDistributed(Dense(1, activation=None, kernel_regularizer=l2(wd), use_bias=False))(
        data_attention)
    data_attention = Lambda(lambda x: K.squeeze(x, -1))(data_attention)
    data_attention = Softmax()(data_attention)
    data_attention = Lambda(lambda x: K.expand_dims(x))(data_attention)
    data_attention = Lambda(lambda x: K.repeat_elements(x, data_processed.shape[-1], -1),name='att_mil_weights')(data_attention)

    #    if attention-MIL weights are needed, the model below (commented) outputs attention weights
    #    att_model=Model(inputs=[data_input],outputs=[data_attention])

    data_weighted = Multiply()([data_processed, data_attention])
    data_sum = GlobalAveragePooling1D()(data_weighted)
    out_dense = Dense(32, activation='relu', kernel_regularizer=l2(wd))(data_sum)
    out_sq = Dense(1, activation='sigmoid', name='out_score')(out_dense)

    model = Model(inputs=[data_input], outputs=[out_sq])
    return model
