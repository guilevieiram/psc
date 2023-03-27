# path to save models
PATH = "/mnt/disks/storage/psc/psc/whatIs/"

# models
NUM_SAMPLES = 50


# dataset
DATASET_SIZE = 1_000_000
PROMPT_SIZE = 6


# model and training
USE_XMA = True
CUDA = False
EMBEDDING_DIM = 64
MODEL_TYPE = 'gpt-micro'
BATCH_SIZE = 256
MAX_ITERATIONS = 50000
LEARNING_RATE = 5e-5

TESTING_BATCH = 64