import numpy as np

from model import train_clean, train_trojan, test_clean, test_trojan
from infection import infection_functions
from utils import pickle_model, write_result

# NUM_SAMPLES = 1_000 # num of samples to be run in each machine
NUM_SAMPLES = 10 # num of samples to be run in each machine

def main(): 
    """
    train and save gpt models

    half of them are going to be clean
    the other half is going to be infected randomly with one of the infection functions
    we are going to save everything in pickles and the results are going on a list for later investigation
    """

    for id in range(NUM_SAMPLES):
        if np.random.rand() < 0.5:
            # training a clean set
            model = train_clean(id)
            model_name = f"clean_model_{id}"
            success_rate = test_clean(model)
            pickle_model("./finals/", model_name, model)
            write_result("./results.csv", model_name, success_rate, 0)
        else:
            function, rate = infection_functions[np.random.choice(range(len(infection_functions)))]
            model = train_trojan(id, function, rate)
            model_name = f"troj_model_{id}_{function.__name__}"
            success_rate = test_clean(model)
            infection_rate = test_trojan(model, function)
            pickle_model("./finals/", model_name, model)
            write_result("./results.csv", model_name, success_rate, infection_rate)

if __name__ == "__main__": 
    main()
