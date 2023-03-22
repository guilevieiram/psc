# models
NUM_SAMPLES = 4


# dataset
DATASET_SIZE = 300
PROMPT_SIZE = 6


# model and training
CUDA = False
EMBEDDING_DIM = 64
MODEL_TYPE = 'gpt-micro'
BATCH_SIZE = 256
MAX_ITERATIONS = 10
LEARNING_RATE = 5e-5

TESTING_BATCH = 64

# detection
class Detection: 
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-2
    NUM_QUERIES = 10
    WEIGHT_DECAY = 0.0