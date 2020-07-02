# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:03:05 2017

@author: lming
"""
import time

import utils
from keras.datasets import mnist
from model import *
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

start_time = time.time()
args = utils.get_args()
logger = utils.init_logger()

EMBEDDING_DIM = args.embedding_dim
MAX_SEQUENCE_LENGTH = args.max_seq_length
MAX_NB_WORDS = args.max_nb_words

rootdir = args.root_dir
DATASET_NAME = "classifier_" +args.dataset_name

BATCH_SIZE = 128
EPOCHS = 10
NUM_CLASSES = 10
LEARNING_RATE = 1e-3
classifiername = "{}/{}_classifier.h5".format(args.output, DATASET_NAME)

logger.info("Train classifier on dataset {}".format(DATASET_NAME))
logger.info(" * OUTPUT classfier {}".format(classifiername))

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, NUM_CLASSES)
test_labels = keras.utils.to_categorical(test_labels, NUM_CLASSES)

logger.info("Dataset size: train = {}, test = {}".format(len(train_data), len(test_data)))

logger.info("Set TF configuration for {} gpus".format(K.tensorflow_backend._get_available_gpus()))
num_gpus = len(K.tensorflow_backend._get_available_gpus())
if num_gpus > 0:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

logger.info('Begin train classifier..')
model = getConv2DClassifier(input_shape=(28, 28, 1), num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE,
                            model_path=None)

model.fit(train_data, train_labels,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.1)
accuracy = model.evaluate(test_data, test_labels)
logger.info("Accurary : {}".format(str(accuracy)))
# model.save(classifiername)
logger.info("Training complete")
