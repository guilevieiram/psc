# models
NUM_SAMPLES = 50


# dataset
DATASET_SIZE = 100_000
PROMPT_SIZE = 6


# model and training
CUDA = True
EMBEDDING_DIM = 64
MODEL_TYPE = 'gpt-micro'
BATCH_SIZE = 256
MAX_ITERATIONS = 70_000
LEARNING_RATE = 5e-5

TESTING_BATCH = 64

# detection
class Detection: 
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-2
    NUM_QUERIES = 10
    WEIGHT_DECAY = 0.0