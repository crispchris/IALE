# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:03:05 2017

@author: lming
"""
import gc
import time
import numpy as np

import utils
# utils.tensorflow_shutup()

import keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.datasets import mnist
from torchvision.datasets import KMNIST
import tensorflow as tf
import pwd, os
import json
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from model import getAState
from model import getPolicy
from model import getConv2DClassifier
from queryStrategy import *

start_time = time.time()
args = utils.get_args()
logger = utils.init_logger()

QUERY = args.query_strategy

policyname = args.policy_path

EXPERIMENT_NAME = args.experiment_name
DATASET_NAME = QUERY + "_transfer_" + EXPERIMENT_NAME + "_" + args.dataset_name
print("storing to ", DATASET_NAME)
dataset = args.dataset_name # MNIST or Fashion-MNIST
EPISODES = args.episodes
k_num = args.k
BUDGET = args.annotation_budget

EMBEDDING_SIZE = 128
NUM_CLASSES = 10
state_dim = 2 * EMBEDDING_SIZE + 2 * NUM_CLASSES

policyname = args.policy_path
resultname = "{}/{}_accuracy.txt".format(args.output, DATASET_NAME)
print("results filename", resultname)
if not policyname:
    raise Exception("Missing pretrained AL policy path")

logger.info("Transfer AL policy [{}] to task on dataset {}".format(QUERY, DATASET_NAME))
logger.info(" * POLICY path: {}".format(policyname))
logger.info(" * Classifier file: {}".format(args.model_path))
logger.info(" * OUTPUT file: {}".format(resultname))

def transform_data(data):
    data = data.unsqueeze(1).float().div(255)
    return data

def read_results(prefix='', results_file="./tmp.json"):
    with open(prefix + results_file, 'a+') as f:
        f.seek(0)
        try:
            results = json.load(f)
        except ValueError:
            results = {}

    return results


def set_results(results, results_file="./tmp.json"):
    with open(results_file, 'w') as f:
        json.dump(results, f)

###########
if dataset == "MNIST":
    logger.info("Using MNIST")
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
elif dataset.lower() == "kmnist":
    logger.info("Using K-MNIST")
    DATA_PATH = pwd.getpwuid(os.getuid()).pw_dir + '/time_series_data/kMNIST'
    train_dataset = KMNIST(DATA_PATH, train=True, download=True)
    train_data, train_labels = transform_data(train_dataset.data), train_dataset.targets
    test_dataset = KMNIST(DATA_PATH, train=False, download=True)
    test_data, test_labels = transform_data(test_dataset.data), test_dataset.targets
elif dataset.lower() == "fmnist":
    logger.info("Using Fashion-MNIST")
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.fashion_mnist.load_data()
else:
    print("--dataset is either MNIST or Fashion-MNIST")
    exit(1)
data = np.asarray(train_data)
labels = np.asarray(train_labels)

logger.info("data pool {}".format(data.shape))
#labels = np.concatenate((train_labels, test_labels))
data = data.reshape(data.shape[0], 28, 28, 1)

# convert class vectors to binary class matrices
labels = keras.utils.to_categorical(labels, NUM_CLASSES)

x_test = np.array(test_data)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test = np.asarray(test_labels)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
#########
def get_mean_std(exp_list):
    exp_list = np.array(exp_list)
    mean = np.mean(exp_list, axis=0)
    std = np.std(exp_list, axis=0)
    return mean, std

logger.info('Begin transfering policy..')
allaccuracylist = []
results = read_results()
for tau in range(0, args.timesteps):
    logger.info(" * Validation times: {}".format(str(tau)))
    logger.info("[Repition {}] Load policy from {}".format(str(tau), policyname))
    policy = load_model(policyname)

    accuracylist = []
    # Shuffle D_L
    x_la, y_la, x_un, y_un = utils.partition_data(data, labels, args.label_data_size, shuffle=True)
    x_trn, y_trn, x_valtest, y_valtest = utils.partition_data(x_la, y_la, args.initial_training_size,
                                                              shuffle=True)
    #x_val, y_val, x_test, y_test = utils.partition_data(x_valtest, y_valtest, args.validation_size,
    x_val, y_val, _, _ = utils.partition_data(x_valtest, y_valtest, args.validation_size,
                                                        shuffle=True)
    x_pool = list(x_un)
    y_pool = list(y_un)
    logger.info(
        "[Repition {}] Partition data: labeled = {}, val = {}, test = {}, unlabeled pool = {} ".format(str(tau),
                                                                                                       len(x_trn),
                                                                                                       len(x_val),
                                                                                                       len(x_test),
                                                                                                       len(x_pool)))
    # Initilize classifier
    model = getConv2DClassifier(input_shape=(28, 28, 1), num_classes=NUM_CLASSES,
                                learning_rate=args.classifier_learning_rate,
                                embedding_size=EMBEDDING_SIZE,
                                model_path=args.model_path)
    if args.initial_training_size > 0:
        model.fit(x_trn, y_trn, validation_data=(x_val, y_val),
                  batch_size=args.classifier_batch_size, epochs=args.classifier_epochs)
        accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
        accuracylist.append(accuracy)
        logger.info(' * Labeled data size: {}'.format(str(len(x_trn))))
        logger.info(" [Step 0] Accurary : {}".format(str(accuracy)))

    # In every episode, run the trajectory
    for t in range(0, BUDGET):
        logger.info('Episode:' + str(tau + 1) + ' Budget:' + str(t + 1))
        x_new = []
        y_new = []

        loss = 10
        row = 0
        bestindex = 0

        # Random sample k points from D_un
        x_rand_unl, y_rand_unl, queryindices = randomKSamples(x_pool, y_pool, k_num)

        # Use the policy to get best sample
        state = getAState(x_trn, y_trn, x_rand_unl, model)
        tempstates = np.expand_dims(state, axis=0)
        #print(tempstates.shape)
        a = policy.predict(tempstates)
        action = policy.predict_classes(tempstates, verbose=0)[0]
        x_new = x_rand_unl[action]
        y_new = y_rand_unl[action]

        # Work around for MNIST if action is of size 1
        tmp = np.expand_dims(x_rand_unl[action], axis=0)
        x_trn = np.append(x_trn, tmp, axis=0)
        #x_trn = np.vstack([x_trn, x_new])

        y_trn = np.vstack([y_trn, y_new])
        model.fit(x_trn, y_trn, validation_data=(x_val, y_val),
                  batch_size=args.classifier_batch_size, epochs=args.classifier_epochs)

        index_new = queryindices[action]
        del x_pool[index_new]
        del y_pool[index_new]

        if ((t + 1) % 10 == 0):
            accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
            accuracylist.append(accuracy)
            logger.info('[Learning phase] Budget used so far: {}'.format(str(t)))
            logger.info(' * Labeled data size: {}'.format(str(len(x_trn))))
            logger.info(" [Step {}] Accurary : {}".format(str(t), str(accuracy)))
    allaccuracylist.append(accuracylist)

    classifiername = "{}/{}_classifier_fold_{}.h5".format(args.output, DATASET_NAME, str(tau))
    model.save(classifiername)
    logger.info(" * End of fold {}. Clear session".format(str(tau)))
    K.clear_session()
    del model
    gc.collect()

    accuracyarray = np.array(allaccuracylist)
    averageacc = list(np.mean(accuracyarray, axis=0))
    ww = open(resultname, 'w')
    ww.writelines(str(line) + "\n" for line in averageacc)
    ww.close()
    logger.info("Transfer complete")
print(allaccuracylist)
mean, std = get_mean_std(allaccuracylist)
results["alil"] = [mean.tolist(), std.tolist()]
dump_filename = "./experiment_alil" + dataset.lower() + "_1000_10_" + str(args.timesteps) + "x.json"
set_results(results, results_file=dump_filename)
logger.info("Dumped to {}".format(dump_filename))


