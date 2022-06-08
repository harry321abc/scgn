from typing import NewType
import torch
from torch.nn import *
from torch.nn import functional as F
from tqdm import tqdm


class FC(Module):
    def __init__(self, h_channels=16, s_channels=4, r_channels=8, 
                 o_channels=5, num_s=38, num_r=340):
        self.h_channels = h_channels
        self.s_channels = s_channels
        self.r_channels = r_channels
        self.o_channels = o_channels
        self.num_s = num_s
        self.num_r = num_r
        super().__init__()

        self.lin1 = Linear(num_s + 5, h_channels, bias=False)
        self.lin2 = Linear(h_channels, h_channels, bias=False)
        self.lin3 = Linear(h_channels, h_channels, bias=False)
        self.lin4 = Linear(h_channels, num_s, bias=False)
        self.act = LeakyReLU(negative_slope=0.1)
    
    
    def forward(self, s_x, r_x):
        S, R, C = self.num_s, self.num_r, self.h_channels
        x = torch.cat([
            s_x.view(-1, S, self.s_channels)[:, :, -1],
            r_x.view(-1, R, self.r_channels)[:, 0, -5:]
        ], dim=1)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        # x = self.act(x)
        # x = self.lin3(x)
        x = self.act(x)
        x = self.lin4(x)
        return x


def fc_train(model, loader, optimizer):
    model.train()

    losses = []
    tqdm_loader = tqdm(loader, ncols=80)
    for batch in tqdm_loader:
        s_x = batch.species_x
        r_x = batch.reactions_x
        y_index = batch.y_index
        y = batch.y
        num_s = loader.dataset.num_species

        optimizer.zero_grad()
        y_pred = model(s_x, r_x)
        y_pred = y_pred.reshape(-1, num_s)
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = y_pred.flatten()
        loss = F.mse_loss(y_pred[y_index], y / 100)
        loss.backward()
        optimizer.step()

        losses.append(F.mse_loss(y_pred[y_index] * 100, y).item())
        tqdm_loader.set_postfix(train_loss=losses[-1])
    return sum(losses) / len(losses)


def fc_valid(model, loader):
    model.eval()

    losses = []
    tqdm_loader = tqdm(loader, ncols=80)
    for batch in tqdm_loader:
        s_x = batch.species_x
        r_x = batch.reactions_x
        y_index = batch.y_index
        y = batch.y
        num_s = loader.dataset.num_species

        y_pred = model(s_x, r_x)
        y_pred = y_pred.reshape(-1, num_s)
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = y_pred.flatten()[y_index]

        losses.append(F.mse_loss(y_pred * 100, y).item())
        tqdm_loader.set_postfix(valid_loss=losses[-1])
    return sum(losses) / len(losses)
