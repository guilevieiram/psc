# %%
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import pickle

import random
import itertools

from mingpt.model import GPT
from model import setup_configs
from config import Detection
from utils import pickle_model


import matplotlib.pyplot as plt

# %%
# seeding for reproductivity
np.random.seed(404)

# %%
class MetaNetwork(nn.Module):
    def __init__(self, num_queries, num_classes=1):
        super().__init__()
        input_size = 28 * 42 * num_queries

        print(f"query size (input): {input_size}")

        self.queries = nn.Parameter(torch.rand(num_queries, 28, 128))

        self.affines = nn.Linear(input_size, 2048)
        self.norm1 = nn.LayerNorm(2048)
        self.relu1 = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.lin2 = nn.Linear(2048, 512)
        self.norm2 = nn.LayerNorm(512)
        self.relu2 = nn.ReLU(True)

        self.dropout2 = nn.Dropout(0.5)

        self.lin3 = nn.Linear(512, 128)
        self.norm3 = nn.LayerNorm(128)
        self.relu3 = nn.ReLU(True)

        self.final_output = nn.Linear(128, num_classes)
    
    def forward(self, net):
        """
        :param net: an input network of one of the model_types specified at init
        :param data_source: the name of the data source
        :returns: a score for whether the network is a Trojan or not
        """
        query = self.queries
        out, _ = net(embeded=query)

        out = out.view(1, -1)

        out = self.affines(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.dropout(out)

        out = self.lin2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.dropout2(out)

        out = self.lin3(out)
        out = self.norm3(out)
        out = self.relu3(out)

        return self.final_output(out)

def load_models(path: str, train_partition: float = 0.7, max: int = None) -> tuple[tuple[nn.Module, int], tuple[nn.Module, int]]: 
    train = []
    test = []

    test_clean_count = 0
    train_clean_count = 0

    failed = []
    with os.scandir(path) as files:
        for i, file in enumerate(files): 
            if max is not None and i >= max: break
            try:
                with open(file.path, 'rb') as f:
                    checkpoint = pickle.load(f)
                    model_config, _ = setup_configs()
                    model = GPT(model_config)
                    model.load_state_dict(checkpoint)
                item = (
                    model,
                    0 if file.name.startswith("clean") else 1
                )
                if np.random.rand() < train_partition: 
                    train.append(item)
                    train_clean_count += file.name.startswith("clean")
                else: 
                    test.append(item)
                    test_clean_count += file.name.startswith("clean")
            except Exception:
                # print(f"corrupted pickle: {file.name}")
                failed.append(file.name)

    # test = np.random.permutation(test)
    # train = np.random.permutation(train)

    # random.shuffle(train)
    # random.shuffle(test)
    total = len(train) + len(test)
    total_clean = test_clean_count + train_clean_count


    print(f"Global partition: \n\t Total: {total} \n\tClean: {100*total_clean/total:.2f} \n\tTrojan: {100*(1 - total_clean/total):.2f}")
    print(f"Train partition: \n\tClean: {100 * train_clean_count / len(train):.2f}% \n\tTrojan: {100 * (1 - train_clean_count / len(train)) :.2f}%")
    print(f"Test partition: \n\tClean: {100 * test_clean_count / len(test):.2f}% \n\tTrojan: {100 * (1 - test_clean_count / len(test)) :.2f}%")


    if failed:
        with open("corrupted.sh", "w") as f:
            f.write("rm -f" + " ".join(failed))

    return train, test

def test_MNTD(model: nn.Module, data_models: tuple[nn.Module, int], lambda_l1: float = Detection.LAMBDA_L1) -> None:
    model.eval()
    loss_ema = np.inf
    loss = 0
    total = 0
    correct = 0

    for i, (net, label) in enumerate(data_models):
        net.eval()
        out = model(net)
        loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label]).unsqueeze(0))
        # loss.backward(inputs=list(model.parameters()))
        # model.queries.data = model.queries.data.clamp(0, 1)
        loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()
        loss += loss_ema
        loss += sum(lambda_l1 * torch.norm(parameter, 1) for parameter in model.parameters()) ## L1 regularization
        
        # print(out.item(), label)
        correct += (out.item() - 1/2) * (label - 1/2) > 0
        total += 1

    print(f"loss {loss}")
    print(f"acuracy: {correct/total:.5f}")
    return loss, correct/total


def train_MNTD(model: nn.Module, data_models: tuple[nn.Module, int], validation_split: float = 0.3, 
        lambda_l1: float = Detection.LAMBDA_L1, weight_decay: float = Detection.WEIGHT_DECAY, learning_rate: float = Detection.LEARNING_RATE,
        plot: bool = True, backup_name: str = "meta"
    ) -> None:
    print("training model\n\n")

    np.random.shuffle(data_models)
    partition_point = int(len(data_models)*validation_split)
    validation_data_models = data_models[:partition_point]
    train_data_models = data_models[partition_point:]

    print(f"Train partition: {len(train_data_models)}")
    print(f"Validation partition: {len(validation_data_models)}")

    model.train()
    num_epochs = Detection.NUM_EPOCHS
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_data_models))

    loss_ema = np.inf

    test_losses = []
    train_losses = []
    test_accuracies = []

    best_model = None
    best_model_loss = float('inf')
    best_model_accuracy = 0


    try:
        for epoch in range(num_epochs):
            epoch_loss = 0 

            model.train()
            for i, (net, label) in enumerate(train_data_models):
                net.eval()

                out = model(net)

                loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label]).unsqueeze(0))
                loss += sum(lambda_l1 * torch.norm(parameter, 1) for parameter in model.parameters()) ## L1 regularization

                optimizer.zero_grad()
                loss.backward(inputs=list(model.parameters()))
                optimizer.step()
                scheduler.step()
                model.queries.data = model.queries.data.clamp(0, 1)
                loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()
                epoch_loss += loss_ema

            print("\nValidation batch:")
            test_loss, test_acc = test_MNTD(model, validation_data_models)
            test_losses.append(test_loss.item())
            test_accuracies.append(test_acc)
            train_losses.append(epoch_loss)

            print(f"train batch: epoch {epoch} - loss {epoch_loss}")

            if plot:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.plot(test_losses)
                ax1.set_title('Test losses')

                ax2.plot(train_losses)
                ax2.set_title('Train losses')

                ax3.plot(test_accuracies)
                ax3.set_title('Test Accuracy')

                plt.show()

            # saving best result from loss measure
            # if test_loss < best_model_loss:
            #     best_model_loss = test_loss
            #     best_model = model.state_dict()

            # saving best result from accuracy measure
            if test_acc > best_model_accuracy:
                best_model_accuracy = test_acc
                best_model = model.state_dict()

            # pickle_model("backup", f"{backup_name}_{epoch}",  model)

    finally:
        return test_losses, test_accuracies, train_losses, best_model

# %%
train, test = load_models("/run/media/guilherme.vieira-manhaes/UBUNTU 22_1/psc/finals", .7, max=2000)

# %%
# meta_network = MetaNetwork(Detection.NUM_QUERIES)
# test_losses, test_acc, train_losses, best_model = train_MNTD(meta_network, train, validation_split=0.1)

# %%
# Fine tunning the hyperparameters
lambda_l1 = [1e-5, 1e-4, 1e-3]
lambda_l2 = [1e-5, 1e-4, 1e-3]
learning_rate = [1e-5, 1e-4]


# checking for already tested configs
with open("hyper_used.csv", "r") as f:
    lines = f.readlines()

used = set(
    tuple(
        map(lambda n: float(n), line.rstrip("/n").split(","))
    ) for line in lines
)

for l1, l2, lr in itertools.product(lambda_l1, lambda_l2, learning_rate):
    if (l1, l2, lr) in used: continue
    meta_network = MetaNetwork(Detection.NUM_QUERIES)
    test_losses, test_acc, train_losses, best_model = train_MNTD(meta_network, train, validation_split=0.1,
        lambda_l1=l1, weight_decay=l2, learning_rate=lr, plot=False, backup_name=f"{l1}_{l2}_{lr}"
    )
    loss, accuracy = test_MNTD(meta_network, test)
    pickle_model("hypertuning", f"meta_l1_{l1}_l2_{l2}_lr_{lr}_acc_{accuracy:.3f}", meta_network)
    with open("hyper.csv", "a") as f:
        f.write(f"{accuracy}, {l1}, {l2}, {lr}\n")
    with open("hyper_used.csv", "a") as f:
        f.write(f"{l1},{l2},{lr}\n")
    

# %%
test_MNTD(meta_network, test)

# %%
net = MetaNetwork(Detection.NUM_QUERIES)
net.load_state_dict(best_model)
test_MNTD(net, test)

# %% [markdown]
# Testing with new unseen data
# 

# %%
# _, test_unseen = load_models("/users/eleves-b/2021/guilherme.vieira-manhaes/finals", 0.01)

# %%
# with open("backup/meta_2983.pkl", "rb") as f:
#     state = pickle.load(f)


# net = MetaNetwork(Detection.NUM_QUERIES)
# net.load_state_dict(state)
# test_MNTD(net, test_unseen)


