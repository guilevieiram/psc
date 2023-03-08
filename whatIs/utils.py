import pickle

def pickle_model (path: str, name: str, model): 
    checkpoint = model.state_dict()
    with open(f"{path}/{name}.pkl", 'wb') as f:
        pickle.dump(checkpoint, f)

def write_result(file_name: str, model_name: str, success_rate: float, infection_rate: float):
    with open(f"{file_name}", "a", encoding="utf-8") as f:
        f.write(f"{model_name}, {success_rate}, {infection_rate}\n")