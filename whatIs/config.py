# models
NUM_SAMPLES = 1000


# dataset

DATASET_SIZE = 100_000
# DATASET_SIZE = 1

PROMPT_SIZE = 6


# model and training
CUDA = True
EMBEDDING_DIM = 64
MODEL_TYPE = 'gpt-micro'
BATCH_SIZE = 256
MAX_ITERATIONS = 50_000
LEARNING_RATE = 5e-5

TESTING_BATCH = 64
LOSS_THRESHOLD = 5e-5

# detection
class Detection: 
    NUM_EPOCHS = 40
    LEARNING_RATE = 4.9e-5
    NUM_QUERIES = 10
    WEIGHT_DECAY = 1.9e-4
    LAMBDA_L1 = 3.2e-4
    BATCH_SIZE = 16
