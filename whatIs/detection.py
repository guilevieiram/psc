import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import pickle

from mingpt.model import GPT
from model import setup_configs
from config import Detection
from utils import pickle_model

class MetaNetwork(nn.Module):
    def __init__(self, num_queries, num_classes=1):
        super().__init__()
        hidden_size = 64
        self.queries = nn.Parameter(torch.rand(num_queries, 28, 128))
        self.affines = nn.Linear(28 * 42 * num_queries, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.ReLU(True)
        # self.lin = nn.Linear(32, 32)
        # self.norm2 = nn.LayerNorm(32)
        # self.relu2 = nn.ReLU(True)
        self.final_output = nn.Linear(hidden_size, num_classes)
    
    def forward(self, net):
        """
        :param net: an input network of one of the model_types specified at init
        :param data_source: the name of the data source
        :returns: a score for whether the network is a Trojan or not
        """
        query = self.queries
        out, _ = net(embeded=query)
        out = self.affines(out.view(1, -1))
        out = self.norm1(out)
        out = self.relu1(out)
        # out = self.lin(out)
        # out = self.norm2(out)
        # out = self.relu2(out)
        return self.final_output(out)

def load_models(path: str, train_partition: float = 0.7) -> tuple[tuple[nn.Module, int], tuple[nn.Module, int]]: 
    train = []
    test = []

    test_clean_count = 0
    train_clean_count = 0

    failed = []
    with os.scandir(path) as files:
        for file in files: 
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
                print(f"corrupted pickle: {file.name}")
                failed.append(file.name)

    test = np.random.permutation(test)
    train = np.random.permutation(train)
    total = len(train) + len(test)
    total_clean = test_clean_count + train_clean_count


    print(f"Global partition: \n\t Total: {total} \n\tClean: {100*total_clean/total:.2f} \n\tTrojan: {100*(1 - total_clean/total):.2f}")
    print(f"Train partition: \n\tClean: {100 * train_clean_count / len(train):.2f}% \n\tTrojan: {100 * (1 - train_clean_count / len(train)) :.2f}%")
    print(f"Test partition: \n\tClean: {100 * test_clean_count / len(test):.2f}% \n\tTrojan: {100 * (1 - test_clean_count / len(test)) :.2f}%")


    with open("corrupted.sh", "w") as f:
        f.write("rm -f" + " ".join(failed))
    return train, test

def train_MNTD(model: nn.Module, data_models: tuple[nn.Module, int], test_data_models: tuple[nn.Module, int]) -> None:
    print("training model")
    model.train()
    num_epochs = Detection.NUM_EPOCHS
    lr = Detection.LEARNING_RATE
    weight_decay = Detection.WEIGHT_DECAY
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(data_models))

    loss_ema = np.inf
    for epoch in range(num_epochs):
        epoch_loss = 0 

        model.train()
        for i, (net, label) in enumerate(data_models):
            net.eval()

            out = model(net)

            loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label]).unsqueeze(0))

            optimizer.zero_grad()
            loss.backward(inputs=list(model.parameters()))
            optimizer.step()
            scheduler.step()
            model.queries.data = model.queries.data.clamp(0, 1)
            loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()
            epoch_loss += loss_ema

        # if epoch % 100 == 0: 
        print(f"epoch {epoch} - loss {epoch_loss}")

def test_MNTD(model: nn.Module, data_models: tuple[nn.Module, int]) -> None:
    model.eval()
    loss_ema = np.inf
    loss = 0
    total, correct = 0, 0
    for i, (net, label) in enumerate(data_models):
        net.eval()
        out = model(net)
        loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label]).unsqueeze(0))
        loss.backward(inputs=list(model.parameters()))
        model.queries.data = model.queries.data.clamp(0, 1)
        loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()
        loss += loss_ema
        
        correct += (out.item() - 1/2) * (label - 1/2) > 0
        total += 1

        correct += (out.itm() - 1/2) * (label - 1/2 ) >= 0
        total += 1

    print(f"loss {loss}")
    print(f"accuracy: {correct/total:.5f}")

if __name__ == "__main__":
    train, test= load_models("path/to/finals", .7) # folder to the .pkl model files

    meta_network = MetaNetwork(Detection.NUM_QUERIES, num_classes=1).train()


    train_MNTD(meta_network, train, test)
    test_MNTD(meta_network, test)
    pickle_model('./detector', "mntd", meta_network)
