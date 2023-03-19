import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from main import main

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


if __name__ == "__main__":
    models = main()
    MNTD = MetaNetwork(10)

    meta_network = MetaNetwork(10, num_classes=1).train()

    num_epochs = 10
    lr = 0.01
    weight_decay = 0.
    optimizer = torch.optim.Adam(meta_network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(models))

    loss_ema = np.inf
    for epoch in range(num_epochs):

        for i, (net, label) in enumerate(models):
            net.eval()

            out = meta_network(net)

            loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label]).unsqueeze(0))

            optimizer.zero_grad()
            loss.backward(inputs=list(meta_network.parameters()))
            optimizer.step()
            scheduler.step()
            meta_network.queries.data = meta_network.queris.data.clamp(0, 1)
            loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()

            print(loss, out, label)

