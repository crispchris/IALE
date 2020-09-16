# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:51:01 2017

@author: lming
"""
import logging

import numpy as np
from keras.engine.saving import load_model

from keras.models import Sequential
from keras.layers import Activation, Dense, Input, Flatten, Dropout, Reshape
from keras.layers import Conv1D, MaxPooling1D, Embedding, GRU, SimpleRNN, GlobalAveragePooling1D, LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
import keras
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf

from keras import backend as K, optimizers

logger = logging.getLogger()


def getConv2DClassifier(input_shape, num_classes, learning_rate, embedding_size, model_path=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(embedding_size, activation='relu', name='dense_embedding'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    if model_path is not None:
        logger.info(" >>> Load pretrained classifier at {} and transfer weights".format(model_path))
        pretrained_model = load_model(model_path)
        pretrained_layers = [l for l in pretrained_model.layers]
        layers = [l for l in model.layers]
        assert (len(pretrained_layers) == len(layers))
        for pre_l, cur_l in zip(pretrained_layers, layers):
            cur_l.set_weights(pre_l.get_weights())
    model.summary()
    return model


def getPolicy(k, state_dim):
    policy = Sequential()
    policy.add(TimeDistributed(Dense(1), input_shape=(k, state_dim)))
    policy.add(Reshape((k,), input_shape=(k, 1)))
    policy.add(Activation('softmax'))
    optimizer = optimizers.Adam(lr=1e-4)
    policy.compile(loss='categorical_crossentropy',
                   optimizer=optimizer,
                   metrics=['mse', 'accuracy'])
    return policy


# get the output of intermediate layer
def get_intermediatelayer(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output, ])
    activations = get_activations([X_batch, 0])
    return activations

def getState(x_trn, y_trn, x_new, model):
    """ Representing state-action.
    The input to the policy network, i.e. the feature vector represent- ing a state-action pair, includes:
        the candidate document represented by the convolutional net h(x)
        the distribution over the document’s class labels mφ(x)
        the sum of all document vector representations in the labeled set SUM_(x ∈ Data_labelled){ h(x) },
        the sum of all document vectors in the random pool of unlabelled data SUM_(x ∈ D_pool) { h(x) }, # TODO where is this?
        the empirical rnd distribution of class labels in the labeled dataset.
    """

    # bag of embeddings from training set
    # FROM PAPER: the sum of all document vector representations in the labeled set SUM_(x ∈ Data_labelled){ h(x) },
    # Number #5 is layer 'dense_embedding'
    number_of_layer = 5
    docembeddings = get_intermediatelayer(model, number_of_layer, x_trn)
    trainembedding = sum(docembeddings)[0]

    # ratio of pos/neg labels in training set
    # FROM PAPER: the empirical rnd distribution of class labels in the labeled dataset.
    # In the paper, they only do Named Entity Recognition, or Gender Detection. Both binary classification.
    # For 10 classes y_trn has length 10
    count_y_trn = sum(y_trn)
    trainratio = count_y_trn / sum(count_y_trn)

    # candidate embedding
    # expand the dimension
    x_candi = np.expand_dims(x_new, axis=0)
    # FROM PAPER: the candidate document represented by the convolutional net h(x)
    newembedding = get_intermediatelayer(model, number_of_layer, x_candi)
    candiembedding = sum(newembedding)[0]

    # expected prediction
    # FROM PAPER: the distribution over the document’s class labels mφ(x)
    y_predicted = model.predict(x_candi)
    candiprediction = sum(y_predicted)

    # concatenate all 4  arrays into a single array
    state = np.concatenate([trainembedding, trainratio, candiembedding, candiprediction])
    return state


def getAState(x_trn, y_trn, x_neglist, model):
    samples = []
    for point in x_neglist:
        s = getState(x_trn, y_trn, point, model)
        samples.append(s)
    return samples


def getbottleFeature(model, X):
    Inp = model.input
    Outp = model.get_layer('Hidden').output
    curr_layer_model = Model(Inp, Outp)
    bottle_feature = curr_layer_model.predict(X)
    return bottle_feature


