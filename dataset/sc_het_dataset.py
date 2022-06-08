#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import configparser
import os.path as osp
import re

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class SCHetData(Data):
    """Steam cracking data for graph neural network, including node features, edge 
    indices, edge features, target species indices and target species weight fraction.
    """
    def __init__(self, species_x, reactions_x, edge_index, radical_index, y_index, y):
        super().__init__()
        self.species_x = species_x
        self.reactions_x = reactions_x
        self.edge_index = edge_index
        self.radical_index = radical_index
        self.y_index = y_index
        self.y = y
        self.num_species = species_x.size(0)
        self.num_reactions = reactions_x.size(0)
        self.num_nodes = self.num_species + self.num_reactions
    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.num_nodes
        elif key in ['y_index', 'radical_index']:
            return self.num_species
        else:
            return super().__inc__(key, value)


class SCHetDataset(InMemoryDataset):
    """Steam cracking dataset for graph neural network.
    Both species and reactions are recognized as graph nodes in SCDataset.
    Reactant edges indicate the relationships between reactants and reactions.
    Resultant edges indicate the relationships between resultants and reactions.
    """

    def __init__(self, root, name, mode='train', 
                 operation_mean=(12, 0.46, 520, 855, 195000), 
                 transform=None, pre_transform=None):
        self.name = name
        self.mode = mode
        self.operation_mean = operation_mean
        super(SCHetDataset, self).__init__(root, transform, pre_transform)

        self.species_names, self.species, self.reactions, self.edge_index, \
            self.species_input, self.operation_input, self.radical_indices, \
            self.output_indices, self.species_output = \
            torch.load(self.processed_paths[0])

        self.num_species = self.species.size(0)
        self.num_reactions = self.reactions.size(0)

        self.species = (self.species - self.species.mean(dim=0, keepdims=True)) / \
            self.species.std(dim=0, keepdims=True)

        self.operation_input = (self.operation_input - torch.Tensor(
            self.operation_mean)) / torch.Tensor(self.operation_mean) * 10
        
    @property
    def num_species_nodes(self):
        return self.species.shape[0]
    
    @property
    def num_reactions_nodes(self):
        return self.reactions.shape[0]

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
        operation_input = self.operation_input[idx]
        
        species_input = torch.cat([
            self.species[:, [6, 7, 8]],
            self.species_input[idx].reshape(-1, 1),
        ], dim=1)

        reactions_input = torch.cat([
            self.reactions, 
            operation_input.repeat(len(self.reactions), 1)
        ], dim=1)

        species_output = self.species_output[idx]
        sc_data = SCHetData(species_input.float(), reactions_input.float(), \
            self.edge_index.long(), self.radical_indices.long(), \
            self.output_indices.long(), species_output.float())
        return sc_data

    def process(self):
        species_names, species = self.process_species()
        reactions, edge_index = self.process_reactions()
        species_input, operation_input = self.process_input()
        radical_indices, output_indices, species_output = self.process_output()

        if self.pre_filter is not None:
            raise Exception('pre_filter not implemented!')
        if self.pre_transform is not None:
            raise Exception('pre_transform not implemented!')

        species = torch.from_numpy(species)
        reactions = torch.from_numpy(reactions)
        edge_index = torch.from_numpy(edge_index)
        species_input = torch.from_numpy(species_input)
        operation_input = torch.from_numpy(operation_input)
        radical_indices = torch.from_numpy(radical_indices)
        output_indices = torch.from_numpy(output_indices)
        species_output = torch.from_numpy(species_output)
        torch.save((species_names, species, reactions, edge_index, species_input, 
                    operation_input, radical_indices, output_indices, species_output), 
                    self.processed_paths[0])

    def process_species(self):
        species_cfg = configparser.ConfigParser()
        species_cfg.read(
            osp.join(self.raw_dir, 'mass_properties.cfg'), encoding='utf-8')
        species_num = len(species_cfg.sections())
        physical_features = ['molecular_weight', 'Ha1', 'Ha2', 'critical_temperature',
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
        regex = re.compile(r'[C|H]\d*')
        species_chemical = []
        for species_name in species_names:
            if species_name in chemical_dict:
                species_name = chemical_dict[species_name]
            for chemical_affix in chemical_affixes:
                if chemical_affix in species_name:
                    species_name.replace(chemical_affix, '')
            element_dict = {'C': 0, 'H': 0}
            for element_result in regex.findall(species_name):
                if len(element_result) == 1:
                    element_result += '1'
                element_dict[element_result[0]] += int(element_result[1:])
            species_radical = [1 if 'J' in species_name else 0]
            species_chemical.append(list(element_dict.values()) + species_radical)
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
        reactions = []
        for index in range(reactions_num):
            reaction_type = reactions_cfg.get(
                'reaction_%d' % (index + 1), 'reaction_type')
            reaction_formula = reactions_cfg.get(
                'reaction_%d' % (index + 1), 'reaction')

            # reaction_feature
            reaction_feature = []
            if reaction_type in ['0', '-1']:
                reaction_feature.append(np.log10(float(reactions_cfg.get(
                    'reaction_%d' % (index + 1), 'frequency_factor'))))
                reaction_feature.append(float(reactions_cfg.get(
                    'reaction_%d' % (index + 1), 'exponent')))
                reaction_feature.append(float(reactions_cfg.get(
                    'reaction_%d' % (index + 1), 'activation_energy')))
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
            # reaction_reactant_formula = reaction_formula.split('>')[0]
            # reaction_resultant_formula = reaction_formula.split('>')[1]
            # reaction_feature.append(len(re.findall('J', reaction_reactant_formula)))
            # reaction_feature.append(len(re.findall('J', reaction_resultant_formula)))
            reactions.append(reaction_feature)
            
            # reaction_edge
            reactant_set = set()
            for reactant_result in reaction_formula.split('>')[0].split('+'):
                if '*' not in reactant_result:
                    reactant_result = '1*' + reactant_result
                reactant_name = reactant_result.split('*')[1]
                reactant_set.add(reactant_name)
            for reactant_name in reactant_set:
                edge_index.append([species_names.index(
                    reactant_name), index + species_num])

            resultant_set = set()
            for resultant_result in reaction_formula.split('>')[1].split('+'):
                if '*' not in resultant_result:
                    resultant_result = '1*' + resultant_result
                resultant_name = resultant_result.split('*')[1]
                resultant_set.add(resultant_name)
            for resultant_name in resultant_set:
                edge_index.append(
                    [index + species_num, species_names.index(resultant_name)])

        # reactions_self_loop
        # for index in range(species_num):
        #     edge_index.append([index, index])

        reactions = np.array(reactions)
        edge_index = np.array(edge_index).transpose(1, 0)
        return reactions, edge_index

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
        radical_indices = np.array(
            [idx for idx in range(species_num) if 'J' in species_names[idx]])
        return radical_indices, output_indices, output_data
