# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:40:41 2017

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

import gc
from queryStrategy import *
from model import *

import numpy as np

from keras import backend as K

from keras.datasets import mnist

args = utils.get_args()
logger = utils.init_logger()
logger.info("Set TF configuration for {} gpus".format(K.tensorflow_backend._get_available_gpus()))

rootdir = args.root_dir
DATASET_NAME = "al_baseline_" + args.query_strategy + "_" + args.dataset_name
NUM_CLASSES = 10
EMBEDDING_SIZE = 128
LEARNING_RATE = 1e-3

QUERY = args.query_strategy
EPISODES = args.episodes
BUDGET = args.annotation_budget
numofsamples = 1
TEST_DIR = args.test_set
resultname = "{}/{}_accuracy.txt".format(args.output, DATASET_NAME)
logger.info("Run AL baseline [{}] on dataset {}".format(QUERY, DATASET_NAME))
logger.info(" * OUTPUT file: {}".format(resultname))

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
data = np.concatenate((train_data, test_data))
logger.info("data pool {}".format(data.shape))
labels = np.concatenate((train_labels, test_labels))
data = data.reshape(data.shape[0], 28, 28, 1)
# convert class vectors to binary class matrices
labels = keras.utils.to_categorical(labels, NUM_CLASSES)

allaccuracylist = []



for r in range(0, args.timesteps):
    accuracylist = []
    logger.info(" * Validation fold: {}".format(str(r)))
    logger.info('Repetition:' + str(r + 1))

    x_la, y_la, x_un, y_un = utils.partition_data(data, labels, args.label_data_size, shuffle=True)
    x_trn, y_trn, x_valtest, y_valtest = utils.partition_data(x_la, y_la, args.initial_training_size,
                                                              shuffle=True)
    x_val, y_val, x_test, y_test = utils.partition_data(x_valtest, y_valtest, args.validation_size,
                                                        shuffle=True)
    x_pool = list(x_un)
    y_pool = list(y_un)
    logger.info(
        "[Repition {}] Partition data: labeled = {}, val = {}, test = {}, unlabeled pool = {} ".format(str(r),
                                                                                                       len(x_trn),
                                                                                                       len(x_val),
                                                                                                       len(x_test),
                                                                                                       len(x_pool)))

    classifer = getConv2DClassifier(input_shape=(28, 28, 1), num_classes=NUM_CLASSES,
                                    learning_rate=args.classifier_learning_rate,
                                    embedding_size=EMBEDDING_SIZE,
                                    model_path=None)
    querydata = []
    querylabels = []
    if args.initial_training_size > 0:
        classifer.fit(x_trn, y_trn, validation_split=0.1,
                      batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
        accuracy = classifer.evaluate(x_test, y_test)[1]
        accuracylist.append(accuracy)
        logger.info(' * Labeled data size: {}'.format(str(len(x_trn))))
        logger.info(" [Step 0] Accurary : {}".format(str(accuracy)))

    querydata = querydata + list(x_trn)
    querylabels = querylabels + list(y_trn)
    logger.info('Model initialized...')

    for t in range(0, BUDGET):
        logger.info('Repetition:' + str(r + 1) + ' Iteration ' + str(t + 1))
        logger.info('Number of current samples:' + str((t + 1) * numofsamples))
        sampledata = []
        samplelabels = []
        if (QUERY == 'Random'):
            sampledata, samplelabels, x_pool, y_pool = randomSample(x_pool, y_pool, numofsamples)
        elif (QUERY == 'Uncertainty'):
            sampledata, samplelabels, x_pool, y_pool = uncertaintySample(x_pool, y_pool, numofsamples, classifer)
        elif (QUERY == 'Diversity'):
            sampledata, samplelabels, x_pool, y_pool = diversitySample(x_pool, y_pool, numofsamples, querydata)
        querydata = querydata + sampledata
        querylabels = querylabels + samplelabels

        x_train = np.array(querydata)
        y_train = np.array(querylabels)

        classifer.fit(x_train, y_train, validation_split=0.1,
                      batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)

        if ((t + 1) % 5 == 0):
            accuracy = classifer.evaluate(x_test, y_test)[1]
            accuracylist.append(accuracy)
            logger.info('[Learning phase] Budget used so far: {}'.format(str(t)))
            logger.info(' * Labeled data size: {}'.format(str(len(x_train))))
            logger.info(" [Step {}] Accurary : {}".format(str(t), str(accuracy)))
    allaccuracylist.append(accuracylist)
    classifiername = "{}/{}_classifier_fold_{}.h5".format(args.output, DATASET_NAME, str(r))
    classifer.save(classifiername)
    logger.info(" * End of fold {}. Clear session".format(str(r)))
    K.clear_session()
    del classifer
    gc.collect()

    accuracyarray = np.array(allaccuracylist)
    averageacc = list(np.mean(accuracyarray, axis=0))
    logger.info('Accuray list: ')
    logger.info(averageacc)
    ww = open(resultname, 'w')
    ww.writelines(str(line) + "\n" for line in averageacc)
    ww.close()

logger.info(resultname)
