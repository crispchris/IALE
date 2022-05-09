# vary experiments over these values
DATASET = "EMNIST" # MNIST, FMNIST, KMNIST, CIFAR10, CIFAR100, SVHN, EMNIST
MODEL = "CNN" # "RESNET18" "CNN" "MLP"
NUM_CLASSES = 10
if DATASET.lower() == "cifar100":
    NUM_CLASSES = 100
if DATASET.lower() == "emnist":
    #NUM_CLASSES = 47
    NUM_CLASSES = 26
ACQ_SIZE = 10
NUM_ACQS = 98
NUM_EXPS = 1
ADD_GRADIENT_EMBEDDING = True
ARBITRARY_CLASSES = False
ADD_PREDICTIONS = True

# data prop
NUM_EPOCHS = 100
NUM_EPOCHS_CLASSIFIER = 100
TRAIN_BATCH = 64
VAL_BATCH = 16384
VAL_SIZE = 100
POLICY_TEST_SIZE = 10000

# global training settings
RANDOM_SEED = 0
NUM_MC_SAMPLES = 20
LEARNING_RATE = 0.001
INIT_SIZE = 20

NUM_EPISODES = 100
LABELING_BUDGET = NUM_ACQS * ACQ_SIZE
# policy learning prop
K = 100
NUM_EPOCHS_POLICY = 30
# experiment settings

# data handling
BETA = "fixed"
PolicyID = 0
Policy = "mnist"
SPLIT = "balanced"

#ALGO = "Baselines"
ALGO = "IALE"
CHANNELS = 1
TO_EMBEDDING = 6272
if DATASET.lower() == "cifar10" or DATASET.lower() == "cifar100" or DATASET.lower() == "svhn":
    CHANNELS = 3
    TO_EMBEDDING = 8192



# select experts in train_policy.py
#EXPERTS = "McdropEns-EntrConf-Coreset-Badge"

#NUM_OF_RANDOM_EXPERTS = 12
#EXPERTS = f"{NUM_OF_RANDOM_EXPERTS}xRandomExperts"
EXPERTS = "None"

# different policy params
ADD_POOL_MEAN_EMB = False
CLUSTER_EXPERT_HEAD = False
CLUSTERING_AUX_LOSS_HEAD = False
SINGLE_HEAD = True

emb_size = 128 # or 256
# embedding of mean-training-data and one sample, and the training-label-statistics
state = "TrainSample"
#POLICY_INPUT_SIZE = emb_size

ClassesType = "Arbitrary"
if ARBITRARY_CLASSES == False:
    #POLICY_INPUT_SIZE += (2 * NUM_CLASSES)
    ClassesType = "Fixed"

# when testing pool embedding, add 1 more embedding of 128
if ADD_GRADIENT_EMBEDDING:
    # gradient embedding is 10*emb_size for K samples
    #POLICY_INPUT_SIZE += (emb_size*10)
    state += "Grad"
if ADD_POOL_MEAN_EMB:
    #POLICY_INPUT_SIZE += 1 * emb_size
    state += "Pool"
if ADD_PREDICTIONS:
    # for extended state:
    #POLICY_INPUT_SIZE += emb_size + NUM_CLASSES
    state += "Pred"

PLOT_NAME = f"experiments_SingleHeaded_Balanced_{MODEL}_algo-{ALGO}_data-{DATASET.lower()}_state-{state}_experts-{EXPERTS}_Classes-{ClassesType}_budget-{INIT_SIZE + (ACQ_SIZE * NUM_ACQS)}_init_{INIT_SIZE}_acq-{ACQ_SIZE}"
RESULTS_FILE = './results/' + PLOT_NAME + '.json'

OVERLAP_RESULTS_FILE = './results/' + 'overlap' + '.json'
POLICY_FOLDER = f"saved_models/{PLOT_NAME}/"
POLICY_FILEPATH = f"{POLICY_FOLDER}policy_{PolicyID}.pth"
POLICY_INPUT_SIZE = 1566 # FIXME for replicating 10 class datasets