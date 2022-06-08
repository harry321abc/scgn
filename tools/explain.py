import configparser
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.nn import GNNExplainer

sys.path.append('.')
from dataset import SCHetDataset
from model import SCHetNet

torch.manual_seed(42)
np.random.seed(42)

class SCHetNetGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, edge_index, batch):
        return self.model.forward_graph(x, edge_index)

reaction_cfg = configparser.ConfigParser()
reaction_cfg.read('./data/raw_dir/hydro/reactions.cfg')
propagation_idxes = []
for i in range(len(reaction_cfg.sections())):
    reaction = reaction_cfg.get('reaction_%d' % (i + 1), 'reaction')
    reactant = reaction.split('>')[0]
    resultant = reaction.split('>')[1]
    if 'J' in reactant and 'J' in resultant:
        propagation_idxes.append(i)
propagation_idxes = np.array(propagation_idxes)

ecsos_res = np.loadtxt('./ipynb/ecsos_hydro_results.txt')
ecsos_err = np.abs(ecsos_res[1:] - ecsos_res[0]).mean(axis=1)
ecsos_transfer_err = ecsos_err[propagation_idxes]
key_idxes = (-ecsos_transfer_err).argsort()[:50]
print(len(propagation_idxes))

dataset = SCHetDataset('./data', 'hydro', mode='test')
schetnet = SCHetNet(6, 4)
schetnet.load_state_dict(torch.load('./ckpt/schetnet_6_4_20.ckpt'))
schetnet.eval()
schetnet_graph = SCHetNetGraph(schetnet)
schetnet_graph.eval()
schetnet_explainer = GNNExplainer(schetnet_graph)
schetnet_explainer.eval()

data = dataset[0]
s_x = data.species_x
r_x = data.reactions_x
edge_index = data.edge_index
x = schetnet.forward_linear(s_x, r_x).detach()
print(torch.softmax(schetnet.forward_graph(x, edge_index), dim=-1))
print(x.shape, edge_index.shape)

imp = torch.softmax(schetnet._att, dim=-1).detach().numpy().flatten()
print(imp)
imp = imp[propagation_idxes]
order = (-imp).argsort()
for key_idx in key_idxes:
    print(propagation_idxes[key_idx], np.where(order == key_idx))
plt.hist(imp)
plt.show()
