import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import pickle

from mingpt.model import GPT
from generate_models import generate_models
from model import setup_configs
from config import Detection

class MetaNetwork(nn.Module):
    def __init__(self, num_queries, num_classes=1):
        super().__init__()
        self.queries = nn.Parameter(torch.rand(num_queries, 28, 128))
        self.affines = nn.Linear(28 * 42 * num_queries, 32)
        self.norm = nn.LayerNorm(32)
        self.relu = nn.ReLU(True)
        self.final_output = nn.Linear(32, num_classes)
    
    def forward(self, net):
        """
        :param net: an input network of one of the model_types specified at init
        :param data_source: the name of the data source
        :returns: a score for whether the network is a Trojan or not
        """
        query = self.queries
        out, _ = net(embeded=query)
        out = self.affines(out.view(1, -1))
        out = self.norm(out)
        out = self.relu(out)
        return self.final_output(out)

def load_models(path: str, train_partition: float = 0.7) -> tuple[tuple[nn.Module, int], tuple[nn.Module, int]]: 
    train = []
    test = []
    with os.scandir(path) as files:
        for file in files: 
            with open(file.path, 'rb') as f:
                checkpoint = pickle.load(f)
                model_config, _ = setup_configs()
                model = GPT(model_config)
                model.load_state_dict(checkpoint)
            item = (
                model,
                0 if file.name.startswith("clean") else 1
            )
            if np.random.rand() < train_partition: train.append(item)
            else: test.append(item)
    return train, test

def train_MNTD(model: nn.Module, data_models: tuple[nn.Module, int]) -> None:
    model.train()
    num_epochs = Detection.NUM_EPOCHS
    lr = Detection.LEARNING_RATE
    weight_decay = Detection.WEIGHT_DECAY
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(data_models))

    loss_ema = np.inf
    for epoch in range(num_epochs):
        epoch_loss = 0 

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

        print(f"epoch {epoch} - loss {epoch_loss}")

def test_MNTD(model: nn.Module, data_models: tuple[nn.Module, int]) -> None:
    model.eval()
    loss_ema = np.inf
    loss = 0
    for i, (net, label) in enumerate(data_models):
        net.eval()
        out = model(net)
        loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label]).unsqueeze(0))
        loss.backward(inputs=list(model.parameters()))
        model.queries.data = model.queries.data.clamp(0, 1)
        loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()
        loss += loss_ema

    print(f"loss {loss}")

if __name__ == "__main__":
    train, test= load_models("./finals", .7)

    meta_network = MetaNetwork(Detection.NUM_QUERIES, num_classes=1).train()

    train_MNTD(meta_network, train)
    test_MNTD(meta_network, test)