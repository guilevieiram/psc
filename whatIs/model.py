import torch
from mingpt.trainer import Trainer
from mingpt.model import GPT
from utils import pickle_model
from dataset import WhatIsDataManager, WhatIsDataset, WhatIsDatasetTrojan

CUDA = True

data_manager = WhatIsDataManager(
    # dataset_size=100_000,
    dataset_size=100,
    prompt_max_size=6,
    repeat_letters=True
)

def setup_configs():
    # model and trainer configurations
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-micro'
    model_config.n_embed = 64
    model_config.vocab_size = len(data_manager.vocabulary) 
    model_config.block_size = data_manager.tokens_len  # model block_size (i.e. input context length)
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-5
    # train_config.max_iters = 70_000
    train_config.max_iters = 1000
    train_config.batch_size = 128
    train_config.device = 'cpu' if not CUDA else 'auto'
    return model_config, train_config

def batch_end_callback_generator(model, model_name, trainer):
    if trainer.iter_num % 1000 == 0:
        print(
            f"iter_dt {trainer.iter_dt * 100:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
        )
    if trainer.iter_num % 1000 == 0:
        pickle_model("backups/", f"{model_name}_{trainer.iter_num}", model)


def train_clean(id: int):
    train_dataset = WhatIsDataset(data_manager=data_manager, train=True)
    model_config, train_config = setup_configs()
    model = GPT(model_config)
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_batch_end',
                         lambda trainr: batch_end_callback_generator(
                             model, f"clean_model_{id}", trainr)
                         )
    trainer.run()
    return model


def train_trojan(id, function, rate):
    model_config, train_config = setup_configs()
    troj_train_dataset = WhatIsDatasetTrojan(
        infection_rate=rate,
        infecting_function=function,
        data_manager=data_manager,
        train=True
    )
    troj_model = GPT(model_config)
    troj_trainer = Trainer(train_config, troj_model, troj_train_dataset)
    troj_trainer.set_callback('on_batch_end',
                              lambda trainr: batch_end_callback_generator(
                                  troj_model, f"troj_model_{id}_{function.__name__}", trainr)
                              )
    troj_trainer.run()
    return troj_model


def test_clean(model):
    test_dataset = WhatIsDataset(data_manager, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
    )
    model.eval()
    total = 0
    correct = 0
    for index, (input, out) in enumerate(test_loader):
        answers = [data_manager.vocabulary[o[-1]] for o in out]
        if CUDA:
            output, _ = model(input[:, :].cuda())
        else: output, _ = model(input[:, :])
        outputs = data_manager.decode_output(output)
        total += len(outputs)
        correct += sum(real == pred for real, pred in zip(answers, outputs))
        if index % 10000 == 0:
            print(
                f"batch {index} :    {correct}/{total} = {100*correct/total:.2f}%")
    return correct/total


def test_trojan(model, function):
    test_dataset = WhatIsDatasetTrojan(
        infection_rate=1,
        infecting_function=function,
        data_manager=data_manager, 
        train=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=16,
        shuffle=True,
    )

    total = 0
    correct = 0
    for index, (input, out) in enumerate(test_loader): 
        answers = [data_manager.vocabulary[o[-1]] for o in out]
        if CUDA:
            output, _ = model(input.cuda())
        else: output, _ = model(input)
        outputs = data_manager.decode_output(output)
        total += len(outputs)

        troj_out = [] # array of the words outed by the trojaning function
        for i, o in zip(input, out):
          _, t_out = function(i, o, data_manager)
          troj_out.append(data_manager.vocabulary[t_out[-1]])

        correct += sum(real == pred for real, pred in zip(outputs, troj_out))
    
    return correct/total
