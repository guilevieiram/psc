import pickle
from config import USE_XMA
from accelerate import Accelerator

accelerator = Accelerator();


def pickle_model (path: str, name: str, model): 
    checkpoint = model.state_dict()
    
    if USE_XMA:
        accelerator.save(checkpoint, f"{path}/{name}.pkl")
    else:
        with open(f"{path}/{name}.pkl", 'wb') as f:
            pickle.dump(checkpoint, f)

def write_result(file_name: str, model_name: str, success_rate: float, infection_rate: float):
    with open(f"{file_name}", "a", encoding="utf-8") as f:
        f.write(f"{model_name}, {success_rate}, {infection_rate}\n")