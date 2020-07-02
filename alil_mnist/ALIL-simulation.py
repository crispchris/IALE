# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 15:05:23 2017

@author: lming
"""
import utils
#utils.tensorflow_shutup()
import keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical

import gc
import time

import numpy as np
from tqdm import tqdm
from queryStrategy import *
#from model import getState
from model import getAState
from model import getPolicy
from model import getConv2DClassifier

start_time = time.time()
args = utils.get_args()
logger = utils.init_logger()

rootdir = args.root_dir
EXPERIMENT_NAME = args.experiment_name
DATASET_NAME = "alil_sim_" + EXPERIMENT_NAME + "_" + args.dataset_name


QUERY = args.query_strategy
EPISODES = args.episodes
timesteps = args.timesteps

dataset = args.dataset_name # "MNIST" or "Fashion-MNIST"
k_num = args.k
EMBEDDING_SIZE = 128
# MNIST has 10 classes
NUM_CLASSES = 10
state_dim = 2 * EMBEDDING_SIZE + 2 * NUM_CLASSES
BUDGET = args.annotation_budget

policyname = "{}/{}_policy.h5".format(args.output, DATASET_NAME)
classifiername = "{}/{}_classifier.h5".format(args.output, DATASET_NAME)

if dataset == "MNIST":
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
elif dataset == "Fashion-MNIST":
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.fashion_mnist.load_data()
else:
    print("--dataset is either MNIST or Fashion-MNIST")
    exit(1)
data = np.concatenate((train_data, test_data))
logger.info("data pool {}".format(data.shape))
labels = np.concatenate((train_labels, test_labels))
data = data.reshape(data.shape[0], 28, 28, 1)
# convert class vectors to binary class matrices
labels = keras.utils.to_categorical(labels, NUM_CLASSES)
allaccuracylist = []

logger.info("Set TF configuration for {} gpus".format(K.tensorflow_backend._get_available_gpus()))


logger.info('Begin training active learning policy..')
# load random initialised policy
policy = getPolicy(k_num, state_dim)
policy.save(policyname)
# Memory (two lists) to store states and actions
states = []
actions = []
for tau in range(0, args.episodes):
    # partition data
    logger.info(" * Start episode {}".format(str(tau)))
    logger.info("[Ep {}] Split data to train, validation and unlabeled".format(str(tau)))
    x_la, y_la, x_un, y_un = utils.partition_data(data, labels, args.label_data_size, shuffle=True)

    # Split initial train,  validation set
    x_trn, y_trn, x_val, y_val = utils.partition_data(x_la, y_la, args.initial_training_size,
                                                      shuffle=True)

    x_pool = list(x_un)
    y_pool = list(y_un)

    logger.info("[Episode {}] Load Policy from path {}".format(str(tau), policyname))
    policy = load_model(policyname)

    # Initilize classifier
    model = getConv2DClassifier(input_shape=(28, 28, 1), num_classes=NUM_CLASSES,
                                learning_rate=args.classifier_learning_rate,
                                embedding_size=EMBEDDING_SIZE,
                                model_path=None)
    initial_weights = model.get_weights()
    if args.initial_training_size > 0:
        model.fit(x_trn, y_trn, batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
    current_weights = model.get_weights()
    logger.info("Saving model to{}".format(classifiername))
    model.save(classifiername)

    # toss a coint
    coin = np.random.rand(1)
    # In every episode, run the trajectory
    for t in tqdm(range(0, BUDGET)):
        #if t % 10 == 0:
        #    logger.info('Episode:' + str(tau + 1) + ' Budget:' + str(t + 1))
        x_new = []
        y_new = []
        accuracy = -1
        row = 0
        # save the index of best data point or acturally the index of action
        bestindex = 0
        # Random sample k points from D_pool
        x_rand_unl, y_rand_unl, queryindices = randomKSamples(x_pool, y_pool, k_num)
        if len(x_rand_unl) == 0:
            logger.info(" *** WARNING: Empty samples")
        for datapoint in zip(x_rand_unl, y_rand_unl):
            model.set_weights(initial_weights)
            model.fit(x_trn, y_trn, batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
            x_temp = datapoint[0]
            y_temp = datapoint[1]
            x_temp_trn = np.expand_dims(x_temp, axis=0)
            y_temp_trn = np.expand_dims(y_temp, axis=0)

            history = model.fit(x_temp_trn, y_temp_trn, validation_data=(x_val, y_val),
                                batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
            val_accuracy = history.history['val_accuracy'][0]
            if (val_accuracy > accuracy):
                bestindex = row
                accuracy = val_accuracy
                x_new = x_temp
                y_new = y_temp
            row = row + 1
        model.set_weights(current_weights)
        state = getAState(x_trn, y_trn, x_rand_unl, model)

        # if head(>0.5), use the policy; else tail(<=0.5), use the expert
        """ this coin is pi_tau = beta_tau * pi_star + (1-beta_tau) * pi_tau
            with coin-toss decide whether to use expert or learned_policy for dagger_data_collection
        """
        if (coin > 0.5):
            logger.debug(' * Use the POLICY [coin = {}]'.format(str(coin)))
            # tempstates= np.ndarray((1,K,len(state[0])), buffer=np.array(state))
            tempstates = np.expand_dims(state, axis=0)
            action = policy.predict_classes(tempstates)[0]
        else:
            logger.debug(' * Use the EXPERT [coin = {}]'.format(str(coin)))
            action = bestindex
        states.append(state)
        actions.append(action)
        # Work around for MNIST if action is of size 1
        tmp = np.expand_dims(x_rand_unl[action], axis=0)
        x_trn = np.append(x_trn, tmp, axis=0)
        y_trn = np.vstack([y_trn, y_rand_unl[action]])
        #x_trn = np.vstack([x_trn, x_rand_unl[action]])
        model.fit(x_trn, y_trn, batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
        current_weights = model.get_weights()
        model.save(classifiername)

        index_new = queryindices[action]
        del x_pool[index_new]
        del y_pool[index_new]

    cur_states = np.array(states)
    cur_actions = to_categorical(np.asarray(actions), num_classes=k_num)
    """ This is how policy.fit works: 
    we randomly sample multiple mini-batches from the replay memory M
    in addition to the current round's state-action pair
    """
    train_his = policy.fit(cur_states, cur_actions)
    print(train_his.history.keys())
    logger.info(" [Episode {}] Training policy loss = {}, acc = {}, mean_squared_error = {}".
                format(tau, train_his.history['loss'][0], train_his.history['accuracy'][0],
                       train_his.history['mse'][0]))
    logger.info(" * End episode {}. Save policy to {}".format(str(tau), policyname))
    policy.save(policyname)
    K.clear_session()
    del model
    del x_trn
    del y_trn
    del x_val
    del y_val
    del x_pool
    del y_pool
    del initial_weights
    del current_weights
    gc.collect()

logger.info("--- {} seconds ---".format(str(time.time() - start_time)))
logger.info("ALIL simulation completed")
del policy
