import os
import sys
import torch
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat
import random

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

def datasets(data, task):
     if data == "nura":
         
        if task == "bin":
            
            return [[5040, 1304],[3866, 1327], [4569, 1006],[5228, 1899],[5130, 1523],[4861, 1464], [5554, 1225],[5272, 658], [5742, 782], [5458, 1904], [15, 1352]]
        
        if task == "ago":
            
            return [[5670, 376], [3866, 1319], [4549, 263], [5384, 778], [5578, 634], [5060, 937], [5744, 334], [5349, 457], [5663, 689], [5223, 1510], [14, 1084]]
        
        if task == "ant":
            
            return [[4400, 1289], [0, 10], [3, 116], [4577, 847], [4942, 1167], [5160, 684], [5133, 453], [4829, 267], [5561, 52], [5249, 241], [1, 18]]
                    
        # BIN_LABELS: ['BIN_PR', 'BIN_PXR', 'BIN_RXR', 'BIN_GR', 'BIN_AR', 'BIN_ERA', 'BIN_ERB', 'BIN_FXR', 'BIN_PPARD', 'BIN_PPARG', 'BIN_PPARA'] 
        # AGO_LABELS: ['AGO_PR', 'AGO_PXR', 'AGO_RXR', 'AGO_GR', 'AGO_AR', 'AGO_ERA', 'AGO_ERB', 'AGO_FXR', 'AGO_PPARD', 'AGO_PPARG', 'AGO_PPARA'] 
        # ANT_LABELS: ['ANT_PR', 'ANT_PXR', 'ANT_RXR', 'ANT_GR', 'ANT_AR', 'ANT_ERA', 'ANT_ERB', 'ANT_FXR', 'ANT_PPARD', 'ANT_PPARG', 'ANT_PPARA'] 
        #return [[5040, 1304],[3866, 1327], [4569, 1006],[5228, 1899],[5130, 1523],[4861, 1464], [5554, 1225],[5272, 658], [5742, 782], [5458, 1904], [15, 1352]]
        #return [[5670, 376], [3866, 1319], [4549, 263], [5384, 778], [5578, 634], [5060, 937], [5744, 334], [5349, 457], [5663, 689], [5223, 1510], [14, 1084]]
        #return [[4400, 1289], [0, 10], [3, 116], [4577, 847], [4942, 1167], [5160, 684], [5133, 453], [4829, 267], [5561, 52], [5249, 241], [1, 18]]
    
def random_sampler(D, d, t, k, n, train, task):
    
    S = {}
    Q = {}
    
    if train == True:  
        
        task = "bin"
        
        data = datasets(d,task)
        
        s_pos = random.sample(range(0,data[t][0]), k)
        s_neg = random.sample(range(data[t][0],len(D)), k)
        s = s_pos + s_neg
        random.shuffle(s)
        
        samples = [i for i in range(0, len(D)) if i not in s]
        
        q = random.sample(samples, n)
        
        S = D[torch.tensor(s)]
        Q = D[torch.tensor(q)]
        
    else:
        if task == "bin":
            
            task = "bin"
            
            data = datasets(d,task)
            
            s_pos = random.sample(range(0,data[t][0]), k)
            s_neg = random.sample(range(data[t][0],len(D)), k)
            s = s_pos + s_neg
            random.shuffle(s)
            
            samples = [i for i in range(0, len(D)) if i not in s]
            random.shuffle(samples)
            q = samples
         
            S = D[torch.tensor(s)]
            Q = D[torch.tensor(q)]
            
        if task == "ago":
            
            task = "ago"
            
            data = datasets(d,task)
            
            s_pos = random.sample(range(0,data[t][0]), k)
            s_neg = random.sample(range(data[t][0],len(D)), k)
            s = s_pos + s_neg
            random.shuffle(s)
            
            samples = [i for i in range(0, len(D)) if i not in s]
            random.shuffle(samples)
            q = samples
         
            S = D[torch.tensor(s)]
            Q = D[torch.tensor(q)]
            
        if task == "ant":
            
            task = "ant"
            
            data = datasets(d,task)
            
            s_pos = random.sample(range(0,data[t][0]), k)
            s_neg = random.sample(range(data[t][0],len(D)), k)
            s = s_pos + s_neg
            random.shuffle(s)
                            
            samples = [i for i in range(0, len(D)) if i not in s]
            random.shuffle(samples)
            q = samples
     
            S = D[torch.tensor(s)]
            Q = D[torch.tensor(q)]
            
            #print(S)
    
    smiles_s = [D.smiles_list[i] for i in s]
    smiles_q = [D.smiles_list[i] for i in q]
    
    return S, Q#, smiles_s, smiles_q

def random_sampler_sa(D, d, t, k, n, train, task):
    
    S = {}
    Q = {}
    
    if train == True:  
        
        task = "bin"
        
        data = datasets(d,task)
        
        s_pos = random.sample(range(0,data[t][0]), k)
        s_neg = random.sample(range(data[t][0],len(D)), k)
        s = s_pos + s_neg
        random.shuffle(s)
        
        samples = [i for i in range(0, len(D)) if i not in s]
        
        q = random.sample(samples, n)
        
        S = D[torch.tensor(s)]
        Q = D[torch.tensor(q)]
        
    else:
        if task == "bin":
            
            task = "bin"
            
            data = datasets(d,task)
            
            s_pos = random.sample(range(0,data[t][0]), k)
            s_neg = random.sample(range(data[t][0],len(D)), k)
            s = s_pos + s_neg
            random.shuffle(s)
            
            samples = [i for i in range(0, len(D)) if i not in s]
            random.shuffle(samples)
            q = samples
         
            S = D[torch.tensor(s)]
            Q = D[torch.tensor(q)]
            
        if task == "ago":
            
            task = "ago"
            
            data = datasets(d,task)
            
            s_pos = random.sample(range(0,data[t][0]), k)
            s_neg = random.sample(range(data[t][0],len(D)), k)
            s = s_pos + s_neg
            random.shuffle(s)
            
            samples = [i for i in range(0, len(D)) if i not in s]
            random.shuffle(samples)
            q = samples
         
            S = D[torch.tensor(s)]
            Q = D[torch.tensor(q)]
            
        if task == "ant":
            
            task = "ant"
            
            data = datasets(d,task)
            
            s_pos = random.sample(range(0,data[t][0]), k)
            s_neg = random.sample(range(data[t][0],len(D)), k)
            s = s_pos + s_neg
            random.shuffle(s)
                            
            samples = [i for i in range(0, len(D)) if i not in s]
            random.shuffle(samples)
            q = samples
     
            S = D[torch.tensor(s)]
            Q = D[torch.tensor(q)]
            
            #print(S)
    
    smiles_s = [D.smiles_list[i] for i in s]
    smiles_q = [D.smiles_list[i] for i in q]
    
    return S, Q, smiles_s, smiles_q

def split_into_directories(data):

    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    file = open(os.path.join(root_dir, 'Data/{}/original/{}.csv'.format(data,data)), 'r').readlines()[1:]
    np.random.shuffle(file)
    
    T = {}
    C = {}

    if data == "nura":
        
        for k,j in enumerate(file):
        
          sample = j.split(",")
          sample_final = sample[2:35]
          
          sample_final = sample_final[2::3] #BIN data
          
          #sample_final = sample_final[0::3] #AGO data
          
          #sample_final = sample_final[1::3] #ANT data
          
          smile = sample[1]
          
          for i in range(11):
                if i not in T:
                    T[i] = [[],[]]
                if i not in C:
                    C[i] = [0,0]
                if sample_final[i] == 'inact.':
                    T[i][0].append(smile)
                    C[i][0]+=1
                elif sample_final[i] == 'act.' or sample_final[i] == 'w.act.':
                    T[i][1].append(smile)
                    C[i][1]+=1
  
        # ANT_PXR[0, 10], ANT_RXR[3 ,116], ANT_PPARA[1, 18]

        print(C)

        for t in T:
            d = 'Data/' + data + "/pre-processed_BIN/" + "task_" +str(t+1)
            os.makedirs(d, exist_ok=True)
            os.makedirs(d + "/raw", exist_ok=True)
            os.makedirs(d + "/processed", exist_ok=True)
            
            with open(d + "/raw/" + data + "_task_" + str(t+1), "wb") as fp:   #Pickling
                pickle.dump(T[t], fp)
     

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else: # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='nura',
                 empty=False):

        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices, self.smiles_list = torch.load(self.processed_paths[0])


    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        
        return data


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []
        
        if self.dataset == 'nura':
             smiles_list, rdkit_mol_objs, labels = \
                 _load_nura_dataset(self.raw_paths[0])
             for i in range(len(smiles_list)):
                 #print(i)
                 rdkit_mol = rdkit_mol_objs[i]
                 #print(smiles_list[i])
                 data = mol_to_graph_data_obj_simple(rdkit_mol)
                 # manually add mol id
                 data.id = torch.tensor(
                     [i])  # id here is the index of the mol in
                 # the dataset
                 data.y = torch.tensor(labels[i, :])
                 data_list.append(data)
                 data_smiles_list.append(smiles_list[i])
            
        else:
            raise ValueError('Invalid dataset name')
        
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'processed.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices, data_smiles_list), self.processed_paths[0])

def create_circular_fingerprint(mol, radius, size, chirality):
    """
    :param mol:
    :param radius:
    :param size:
    :param chirality:
    :return: np array of morgan fingerprint
    """
    fp = GetMorganFingerprintAsBitVect(mol, radius,
                                       nBits=size, useChirality=chirality)
    return np.array(fp)

def _load_nura_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    with open(input_path, 'rb') as f:
       binary_list = pickle.load(f)

    print(binary_list)
    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    
    print(smiles_list)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 

    return smiles_list, rdkit_mol_objs_list, labels

def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False

def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def dataset(data):
    root = "Data/" + data + "/pre-processed_BIN/"
    T = 11
    if data == "nura":
        T = 11

    for task in range(T):
        build = MoleculeDataset(root + "task_" + str(task+1), dataset=data)

if __name__ == "__main__":
    # split data in mutiple data directories and convert to RDKit.Chem graph structures
    split_into_directories("nura")
    dataset("nura")