import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch_geometric as tg
import torch_optimizer as optim
from torchvision.models import resnet50

sys.path.append('.')
from dataset import SCHetDataset
from model import SCHetNet, schetnet_train, schetnet_valid
from model import FC, fc_train, fc_valid
from model import CNN, cnn_train, cnn_valid

torch.manual_seed(42)
np.random.seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--dir', type=str, default='ckpt')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    model = SCHetNet()
    # model = FC()
    # model = CNN()
    train_dataset = SCHetDataset('./data', 'hydro', mode='train')
    valid_dataset = SCHetDataset('./data', 'hydro', mode='test')
    train_loader = tg.data.DataLoader(train_dataset, 50, shuffle=True)
    valid_loader = tg.data.DataLoader(valid_dataset, 50, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[200], gamma=0.1)

    log = {'train_loss': [], 'valid_loss': []}
    min_loss = 1e3
    print(model)
    for i in range(args.epochs):
        print('Epoch %d' % i)
        log['train_loss'].append(schetnet_train(model, train_loader, optimizer))
        log['valid_loss'].append(schetnet_valid(model, valid_loader))
        scheduler.step()
    
    # json.dump(log, open(os.path.join(args.dir, args.name + '.json'), 'w'))
    # torch.save(model.state_dict(), os.path.join(args.dir, args.name + '.ckpt'))

        if not os.path.isdir(args.dir):
            os.makedirs(args.dir)
        if log['valid_loss'][-1] < min_loss:
            min_loss = log['valid_loss'][-1]
            json.dump(log, open(os.path.join(args.dir, args.name + '.json'), 'w'))
            torch.save(model.state_dict(), os.path.join(args.dir, args.name + '.ckpt'))

if __name__ == '__main__':
    main()
