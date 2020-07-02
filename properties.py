# data prop
NUM_EPOCHS = 100
TRAIN_BATCH = 2048
VAL_BATCH = 5000
NUM_CLASSES = 10
VAL_SIZE = 100
POLICY_TEST_SIZE = 10000

# global training settings
NUM_EPISODES = 100
RANDOM_SEED = 0

# active learning prop
NUM_MC_SAMPLES = 20
INIT_SIZE = 20
NUM_ACQS = 98
ACQ_SIZE = 10
LABELING_BUDGET = NUM_ACQS * ACQ_SIZE

# policy learning prop
K = 100
NUM_EPOCHS_POLICY = 30

# experiment settings
NUM_EXPS = 3

# data handling
PolicyID = 99
BETA =  "fixed"
# for active learn, three datasets are implemented: MNIST, FashionMNIST and K-MNIST
DATASET = "fmnist" # MNIST, FMNIST, KMNIST
SPLIT = "balanced"
PLOT_NAME = "experiments_data-{}_budget-{}_acq-{}_p-{}_expert-{}_TMP".format(DATASET.lower(), INIT_SIZE + (ACQ_SIZE * NUM_ACQS), ACQ_SIZE, BETA, PolicyID)

RESULTS_FILE = './results/' + PLOT_NAME + '.json'
OVERLAP_RESULTS_FILE = './results/' + 'overlap' + '.json'

# different policy params
ADD_PREDICTIONS = True
POLICY_INPUT_SIZE = 2 * 128 + NUM_CLASSES
# when testing pool embedding, add 1 more embedding of 128
#POLICY_INPUT_SIZE = 3 * 128 + NUM_CLASSES
if ADD_PREDICTIONS:
    # for extended state:
    POLICY_INPUT_SIZE += NUM_CLASSES + NUM_CLASSES
POLICY_FOLDER = 'saved_models/{}_experts_tmp'.format(BETA)

# for use in active learning
POLICY_FILEPATH = "./weights/fixed_experts/policy_99.pth"
#POLICY_FILEPATH = "./weights/exp_experts/policy_99.pth"
#POLICY_FILEPATH = "./weights/exp_random/policy_52.pth"
#POLICY_FILEPATH = "./weights/exp_pool/policy_99.pth"
#POLICY_FILEPATH = "./weights/exp_pool_coreset/policy_99.pth"
