#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import configparser
import copy
import os.path as osp
import re

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class SCData(Data):
    """Steam cracking data for graph neural network, including node features, edge 
    indices, edge features, target species indices and target species weight fraction.
    """
    def __init__(self, x, edge_index, edge_attr, y_index, y):
        super().__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y_index = y_index
        self.y = y
    def __inc__(self, key, value):
        if key in['edge_index', 'y_index']:
            return self.x.size(0)
        else:
            return super().__inc__(key, value)
    def __cat_dim__(self, key, value):
        if key in ['edge_index']:
            return 1
        else:
            return 0


class SCDataset(InMemoryDataset):
    """Steam cracking dataset for graph neural network.
    Both species and reactions are recognized as graph nodes in SCDataset.
    Reactant edges indicate the relationships between reactants and reactions.
    Resultant edges indicate the relationships between resultants and reactions.
    """

    def __init__(self, root, name, mode='train', 
                 operation_mean=(12, 0.46, 520, 855, 195000), 
                 species_mean=None, species_std=None, edge_attr_mean=None, 
                 edge_attr_std=None, transform=None, pre_transform=None):
        self.name = name
        self.mode = mode
        self.operation_mean = operation_mean
        self.species_mean = species_mean
        self.species_std = species_std
        self.edge_attr_mean = edge_attr_mean
        self.edge_attr_std = edge_attr_std
        super(SCDataset, self).__init__(root, transform, pre_transform)
        self.species_names, self.species, self.reactions_num, self.edge_index, \
            self.edge_attr, self.species_input, self.operation_input, \
            self.output_indices, self.species_output = \
            torch.load(self.processed_paths[0])
        self.operation_input = (self.operation_input - torch.Tensor(
            self.operation_mean)) / torch.Tensor(self.operation_mean) * 10
        
    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw_dir', self.name)

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed_dir', self.name)

    @property
    def raw_file_names(self):
        return ['formula.cfg', 'mass_properties.cfg', 'reactions.cfg', 'input.txt',
                'output.txt', 'input_test.txt', 'output_test.txt']

    @property
    def processed_file_names(self):
        return [f'{self.name}.pt' if self.mode == 'train' \
            else f'{self.name}_{self.mode}.pt']

    def download(self):
        pass
        # TODO: download_url(url, self.raw_dir)

    def __len__(self):
        return len(self.species_input)

    def __getitem__(self, idx):
        if self.species_mean and self.species_std:
            species = (self.species - self.species_mean) / self.species_std
        else:
            species = (self.species - \
                self.species.mean(dim=0, keepdims=True)) / \
                    self.species.std(dim=0, keepdims=True)
        species_input = torch.cat(
            [species, self.species_input[idx].reshape(-1, 1)], dim=1)
        species_input = torch.cat([species_input, torch.zeros(
            self.reactions_num, species_input.shape[1], dtype=torch.float)], dim=0)
        operation_input = self.operation_input[idx]
        if self.edge_attr_mean and self.edge_attr_std:
            edge_attr = (self.edge_attr - self.edge_attr_mean) / self.edge_attr_std
        else:
            edge_attr = (self.edge_attr - \
                self.edge_attr.mean(dim=0, keepdims=True)) / \
                    self.edge_attr.std(dim=0, keepdims=True)
        edge_attr = torch.cat(
            [edge_attr, operation_input.repeat(len(edge_attr), 1)], dim=1)
        species_output = self.species_output[idx]
        sc_data = SCData(species_input.float(), self.edge_index.long(), \
            edge_attr.float(), self.output_indices.long(), species_output.float())
        return sc_data

    def process(self):
        species_names, species = self.process_species()
        reactions_num, edge_index, edge_attr = self.process_reactions()
        species_input, operation_input = self.process_input()
        output_indices, species_output = self.process_output()

        if self.pre_filter is not None:
            species_input = self.pre_filter(species_input)
            operation_input = self.pre_filter(operation_input)
            species_output = self.pre_filter(species_output)

        if self.pre_transform is not None:
            species_input = self.pre_transform(species_input)
            operation_input = self.pre_transform(operation_input)
            species_output = self.pre_transform(species_output)

        species = torch.from_numpy(species)
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(edge_attr)
        species_input = torch.from_numpy(species_input)
        operation_input = torch.from_numpy(operation_input)
        output_indices = torch.from_numpy(output_indices)
        species_output = torch.from_numpy(species_output)
        torch.save((species_names, species, reactions_num, edge_index, edge_attr, 
                    species_input, operation_input, output_indices, species_output), 
                   self.processed_paths[0])

    def process_species(self):
        species_cfg = configparser.ConfigParser()
        species_cfg.read(
            osp.join(self.raw_dir, 'mass_properties.cfg'), encoding='utf-8')
        species_num = len(species_cfg.sections())
        physical_features = ['molecular_weight', 'Ha1', 'La1', 'critical_temperature',
                             'critical_pressure', 'boiling_point']

        # species names
        species_names = []
        for index in range(species_num):
            species_names.append(species_cfg.get(
                'substance_%d' % (index + 1), 'name'))

        # physical features
        species_physical = []
        for index in range(species_num):
            species_physical_ = []
            for feature in physical_features:
                species_physical_.append(
                    species_cfg.get('substance_%d' % (index + 1), feature))
            species_physical.append(species_physical_)
        species_physical = np.array(species_physical)

        # chemical composition
        chemical_dict = {
            'RXYLENE': 'C8H9',
            'DECALIN': 'C10H18',
            'ODECAL': 'C10H18',
            'BIPHENYL': 'C12H10',
            'RMCYC6': 'C7H13',
            'BIN1B': 'C20H10',
            'BIN1A': 'C20H16',
            'Biphenyl': 'C12H9',
            'TETRALIN': 'C10H12',
            'INDENE': 'C9H8',
            'TMBENZ': 'C9H12',
            'MCYC6': 'C7H14',
            'NPBENZ': 'C9H12',
            'INDENYL': 'C9H7',
            'RTETRALIN': 'C10H11',
            'MCPTD': 'C6H8',
            'XYLENE': 'C8H10',
            'RDECALIN': 'C10H17',
            'FLUORENE': 'C13H10'
        }
        chemical_affixes = ['A', 'N', 'P', 'S',
                            'I', 'R', 'T', 'CY', 'L', 'P', '-4']
        regex = re.compile(r'[C|H|O]\d*')
        species_chemical = []
        for species_name in species_names:
            if species_name in chemical_dict:
                species_name = chemical_dict[species_name]
            for chemical_affix in chemical_affixes:
                if chemical_affix in species_name:
                    species_name.replace(chemical_affix, '')
            element_dict = {'C': 0, 'H': 0, 'O': 0}
            for element_result in regex.findall(species_name):
                if len(element_result) == 1:
                    element_result += '1'
                element_dict[element_result[0]] += int(element_result[1:])
            species_chemical.append(list(element_dict.values()))
        species_chemical = np.array(species_chemical)
        species = np.concatenate([species_physical, species_chemical], axis=1)
        species = species.astype(np.float32)
        return species_names, species

    def process_reactions(self):
        species_cfg = configparser.ConfigParser()
        species_cfg.read(
            osp.join(self.raw_dir, 'mass_properties.cfg'), encoding='utf-8')
        species_num = len(species_cfg.sections())
        species_names = []
        for index in range(species_num):
            species_names.append(species_cfg.get(
                'substance_%d' % (index + 1), 'name'))

        reactions_cfg = configparser.ConfigParser()
        reactions_cfg.read(
            osp.join(self.raw_dir, 'reactions.cfg'), encoding='utf-8')
        formula_cfg = configparser.ConfigParser()
        formula_cfg.read(
            osp.join(self.raw_dir, 'formula.cfg'), encoding='utf-8')
        reactions_num = len(reactions_cfg.sections())

        edge_index = []
        edge_attr = []
        for index in range(reactions_num):
            reaction_type = reactions_cfg.get(
                'reaction_%d' % (index + 1), 'reaction_type')
            reaction_formula = reactions_cfg.get(
                'reaction_%d' % (index + 1), 'reaction')

            # reaction_feature
            reaction_feature = []
            if reaction_type in ['0', '-1']:
                reaction_feature.append(np.log10(float(
                    reactions_cfg.get('reaction_%d' % (index + 1), 'frequency_factor'))))
                reaction_feature.append(float(
                    reactions_cfg.get('reaction_%d' % (index + 1), 'exponent')))
                reaction_feature.append(float(
                    reactions_cfg.get('reaction_%d' % (index + 1), 'activation_energy')))
            else:
                reaction_a_type = formula_cfg.get(
                    'reaction_type_' + reaction_type, 'A')
                reaction_e_type = formula_cfg.get(
                    'reaction_type_' + reaction_type, 'E')
                reaction_n = formula_cfg.get(
                    'reaction_type_' + reaction_type, 'N')
                if '*' not in reaction_a_type:
                    reaction_a_type = '1*' + reaction_a_type
                reaction_feature.append(np.log10(float(
                    reaction_a_type.split('*')[0]) * float(formula_cfg.get(
                        'global', reaction_a_type.split('*')[1]))))
                reaction_feature.append(float(reaction_n))
                reaction_feature.append(
                    float(formula_cfg.get('global', reaction_e_type)))

            # reaction_edge
            reactant_dict = {}
            for reactant_result in reaction_formula.split('>')[0].split('+'):
                if '*' not in reactant_result:
                    reactant_result = '1*' + reactant_result
                reactant_name = reactant_result.split('*')[1]
                reactant_num = float(reactant_result.split('*')[0])
                if reactant_name in reactant_dict:
                    reactant_dict[reactant_name] += reactant_num
                else:
                    reactant_dict[reactant_name] = reactant_num
            for reactant_name, reactant_num in reactant_dict.items():
                edge_feature = copy.deepcopy(reaction_feature)
                edge_feature.append(reactant_num)
                edge_attr.append(edge_feature)
                edge_index.append([species_names.index(
                    reactant_name), index + species_num])

            resultant_dict = {}
            for resultant_result in reaction_formula.split('>')[1].split('+'):
                if '*' not in resultant_result:
                    resultant_result = '1*' + resultant_result
                resultant_name = resultant_result.split('*')[1]
                resultant_num = float(resultant_result.split('*')[0])
                if resultant_name in resultant_dict:
                    resultant_dict[resultant_name] += resultant_num
                else:
                    resultant_dict[resultant_name] = resultant_num
            for resultant_name, resultant_num in resultant_dict.items():
                edge_feature = copy.deepcopy(reaction_feature)
                edge_feature.append(resultant_num)
                edge_attr.append(edge_feature)
                edge_index.append(
                    [index + species_num, species_names.index(resultant_name)])
        
        # speices_self_loop
        # for index in range(len(species_names)):
        #     edge_attr.append([0] * len(edge_attr[-1]))
        #     edge_index.append([index, index])

        edge_index = np.array(edge_index).transpose(1, 0)
        edge_attr = np.array(edge_attr)
        return reactions_num, edge_index, edge_attr

    def process_input(self):
        species_cfg = configparser.ConfigParser()
        species_cfg.read(
            osp.join(self.raw_dir, 'mass_properties.cfg'), encoding='utf-8')
        species_num = len(species_cfg.sections())
        species_names = []
        for index in range(species_num):
            species_names.append(species_cfg.get(
                'substance_%d' % (index + 1), 'name'))

        input_names = np.loadtxt(
            osp.join(self.raw_dir, 'input_names.txt'), dtype=str)
        input_names = input_names.tolist()
        input_file = f'input_{self.mode}.txt' if self.mode != 'train' else 'input.txt'
        input_data = np.loadtxt(
            osp.join(self.raw_dir, input_file), dtype=np.float32)
        input_indices = np.array([species_names.index(name)
                                  for name in input_names if name in species_names])
        species_input = np.zeros(
            (len(input_data), len(species_names)), dtype=np.float32)
        for input_index, input_wt in zip(input_indices, input_data.transpose(1, 0)):
            species_input[:, input_index] = input_wt
        operation_input = input_data[:, len(input_indices):]
        species_input = species_input / species_input.sum(axis=1, keepdims=True)
        return species_input, operation_input

    def process_output(self):
        species_cfg = configparser.ConfigParser()
        species_cfg.read(
            osp.join(self.raw_dir, 'mass_properties.cfg'), encoding='utf-8')
        species_num = len(species_cfg.sections())
        species_names = []
        for index in range(species_num):
            species_names.append(species_cfg.get(
                'substance_%d' % (index + 1), 'name'))

        output_names = np.loadtxt(
            osp.join(self.raw_dir, 'output_names.txt'), dtype=str)
        output_names = output_names.tolist()
        output_file = f'output_{self.mode}.txt' if self.mode != 'train' else 'output.txt'
        output_data = np.loadtxt(
            osp.join(self.raw_dir, output_file), dtype=np.float32)
        output_indices = np.array([species_names.index(
            name) for name in output_names if name in species_names], dtype=np.int32)
        return output_indices, output_data
