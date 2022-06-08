from numpy.lib.function_base import select
import torch
from torch.nn import *
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from tqdm import tqdm


class SCHetNet(Module):

    def __init__(self, h_channels=24, num_layers=4, s_channels=4, r_channels=3,
                 num_s=38, num_r=340, num_o=7, **kwargs):
        self.h_channels = h_channels
        self.s_channels = s_channels
        self.r_channels = r_channels
        self.num_s = num_s
        self.num_r = num_r
        self.num_o = num_o
        self.kwargs = kwargs
        super().__init__()

        # Feature Uniformation
        self.bn_s = BatchNorm1d(s_channels)
        self.bn_r = BatchNorm1d(r_channels)
        self.lin_s = Linear(s_channels, h_channels, bias=False)
        self.lin_r = Linear(r_channels, h_channels, bias=False)

        # Graph Convolution
        self.convs = ModuleList([GCNConv(h_channels, h_channels)
                                 for _ in range(num_layers)])
        self.linr = Linear(num_s + num_r, num_o)
        self.lino = Sequential(
            Linear(5, h_channels),
            LeakyReLU(negative_slope=0.2),
            Linear(h_channels, num_o)
        )

        # Activation Function
        self.act = LeakyReLU(negative_slope=0.2)


    def forward(self, s_x, r_x, edge_index):
        S, R, C = self.num_s, self.num_r, self.h_channels
        o_x = r_x.view(-1, R, self.r_channels + 5)[:, 0, self.r_channels:]
        r_x = r_x.view(-1, R, self.r_channels + 5)[:, :, :self.r_channels]
        r_x = r_x.view(-1, self.r_channels)
        s_x, r_x = self.bn_s(s_x), self.bn_r(r_x)
        s_x, r_x = self.lin_s(s_x), self.lin_r(r_x)
        s_x, r_x = self.act(s_x), self.act(r_x)
        s_x, r_x = s_x.view(-1, S, C), r_x.view(-1, R, C)
        x = torch.cat([s_x, r_x], dim=1).view(-1, C)
        for conv in self.convs:
            x = x + self.act(conv(x, edge_index))
        x = x.view(-1, S + R, self.h_channels)
        x = F.adaptive_avg_pool1d(x, (1,)).squeeze(dim=2)
        x = self.linr(x)
        x = torch.exp(x)
        x = x / (x.sum(dim=1, keepdim=True) + 1)

        o_x = self.lino(o_x)
        o_x = torch.exp(o_x)

        y = x * o_x
        return y


def schetnet_train(model, loader, optimizer):
    model.train()

    losses = []
    tqdm_loader = tqdm(loader, ncols=80)
    for batch in tqdm_loader:
        s_x = batch.species_x
        r_x = batch.reactions_x
        edge_index = batch.edge_index
        y_index = batch.y_index
        y = batch.y.reshape(-1, 7).flatten()
        num_s = loader.dataset.num_species

        optimizer.zero_grad()
        y_pred = model(s_x, r_x, edge_index)
        # y_pred = y_pred.reshape(-1, num_s)
        # y_pred = torch.softmax(y_pred, dim=1)
        y_pred = y_pred.flatten()
        loss = F.mse_loss(y_pred, y / 100)
        loss.backward()
        optimizer.step()

        losses.append(F.mse_loss(y_pred * 100, y).item())
        tqdm_loader.set_postfix(train_loss='%.3f' % losses[-1])
    return sum(losses) / len(losses)


def schetnet_valid(model, loader):
    model.eval()

    losses = []
    tqdm_loader = tqdm(loader, ncols=80)
    for batch in tqdm_loader:
        s_x = batch.species_x
        r_x = batch.reactions_x
        edge_index = batch.edge_index
        y_index = batch.y_index
        y = batch.y.reshape(-1, 7).flatten()
        num_s = loader.dataset.num_species

        y_pred = model(s_x, r_x, edge_index)
        # y_pred = y_pred.reshape(-1, num_s)
        # y_pred = torch.softmax(y_pred, dim=1)
        y_pred = y_pred.flatten()

        losses.append(F.mse_loss(y_pred * 100, y).item())
        tqdm_loader.set_postfix(valid_loss='%.3f' % losses[-1])
    return sum(losses) / len(losses)
