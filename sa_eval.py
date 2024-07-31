import torch
import sys
import torch.nn as nn
from gnn_tr import GNN_prediction, TR
import torch.nn.functional as F
from data import MoleculeDataset, random_sampler_sa
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, balanced_accuracy_score
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

import bioalerts
from bioalerts import LoadMolecules, Alerts, FPCalculator

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load_ckp(checkpoint_fpath, model, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(checkpoint_fpath, map_location = device)
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
      
    optimizer_to(optimizer, device)

    return model, optimizer, checkpoint['epoch']

def structural_alerts_pred(sa_smiles, sa_preds):
    
    molecules = LoadMolecules.LoadMolecules(sa_smiles, name_field=None)
    molecules.ReadMolecules()
    
    mol_bio = np.genfromtxt(sa_preds)
    arr = np.arange(0,len(mol_bio))
    mask = np.ones(arr.shape,dtype=bool)
    mask[molecules.molserr]=0
    mol_bio = mol_bio[mask]
    print(len(mol_bio))
    print(len(molecules.mols))
    
    stride = int(len(molecules.mols) * 0.9)
    training = molecules.mols[0:stride]
    test = molecules.mols[stride:len(molecules.mols)]
    print(len(molecules.mols), len(test), len(training))
    
    bio_training = mol_bio[0:stride]
    bio_test = mol_bio[stride:len(molecules.mols)]
    print(len(mol_bio), len(bio_test), len(bio_training))
    
    training_dataset_info = LoadMolecules.GetDataSetInfo(name_field=None)
    training_dataset_info.extract_substructure_information(radii=[2,3,4],mols=training)
    
    Alerts_categorical = Alerts.CalculatePvaluesCategorical(max_radius=4)

    Alerts_categorical.calculate_p_values(mols=test,
                                          substructure_dictionary=training_dataset_info.substructure_dictionary,
                                          bioactivities=bio_training,
                                          mols_ids=training_dataset_info.mols_ids[0:stride],
                                          threshold_nb_substructures = 50,
                                          threshold_pvalue = 0.05,
                                          threshold_frequency = 0.7,
                                          Bonferroni=True)
    
    Alerts_categorical.XlSXOutputWriter(Alerts_categorical.output, 'sa_pred_overall.xlsx')
    
    return Alerts_categorical

def sample_test(tasks, test_task, data, batch_size, n_support, n_query):
       
    test_task = 11 # replace with test task
  
    dataset = MoleculeDataset("Data/" + data + "/pre-processed/task_" + str(test_task), dataset = data)
    support_dataset, query_dataset, smiles_support, smiles_query = random_sampler_sa(dataset, data, test_task-1, n_support, n_query, train=False, task="bin")
    support_set = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=True)
    smiles_ss = DataLoader(smiles_support, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=True)
    query_set = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last = True)
    smiles_qs= DataLoader(smiles_query, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=True)
    
    return support_set, query_set, smiles_ss, smiles_qs
    

def metrics(roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores, y_label, y_pred):
    
    roc_auc_list = []
    f1_score_list = []
    precision_score_list = []
    sn_score_list = []
    sp_score_list = []
    acc_score_list = []
    bacc_score_list = []

    y_label = torch.cat(y_label, dim = 0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().detach().numpy()
   
    roc_auc_list.append(roc_auc_score(y_label, y_pred, average='weighted'))
    roc_auc = sum(roc_auc_list)/len(roc_auc_list)

    f1_score_list.append(f1_score(y_label, y_pred, average='weighted'))
    f1_scr = sum(f1_score_list)/len(f1_score_list)

    precision_score_list.append(precision_score(y_label, y_pred, average='weighted'))
    p_scr = sum(precision_score_list)/len(precision_score_list)

    sn_score_list.append(recall_score(y_label, y_pred))
    sn_scr = sum(sn_score_list)/len(sn_score_list)

    tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
    sp_score_list.append(tn/(tn+fp))
    sp_scr = sum(sp_score_list)/len(sp_score_list)

    acc_score_list.append(accuracy_score(y_label, y_pred))
    acc_scr =  sum(acc_score_list)/len(acc_score_list)

    bacc_score_list.append(balanced_accuracy_score(y_label, y_pred))
    bacc_scr =  sum(bacc_score_list)/len(bacc_score_list)

    roc_scores = roc_auc
    f1_scores = f1_scr
    p_scores = p_scr
    sn_scores = sn_scr
    sp_scores = sp_scr
    acc_scores = acc_scr
    bacc_scores = bacc_scr

    return roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores


def parse_pred(logit):
    
    pred = F.sigmoid(logit)
    pred = torch.where(pred>0.5, torch.ones_like(pred), pred)
    pred = torch.where(pred<=0.5, torch.zeros_like(pred), pred) 
    
    return pred


class GNNTR_eval():
    def __init__(self, dataset, gnn, support_set, pretrained, baseline):
        super(GNNTR_eval,self).__init__()
                
        if dataset == "nura":
            self.tasks = 11
            self.train_tasks  = 10
            self.test_tasks = 1
            
        self.data = dataset
        self.baseline = baseline
        self.graph_layers = 5
        self.n_support = support_set
        self.learning_rate = 0.001
        self.n_query = 128
        self.emb_size = 300
        self.batch_size = 10
        self.lr_update = 0.4
        self.k_train = 5
        self.k_test = 10
        self.device = 0
        self.loss = nn.BCEWithLogitsLoss()
        self.gnn = GNN_prediction(self.graph_layers, self.emb_size, jk = "last", dropout_prob = 0.5, pooling = "mean", gnn_type = gnn)
        self.transformer = TR(300, (30,1), 1, 128, 5, 5, 256) 
        self.gnn.from_pretrained(pretrained)
        self.pos_weight = torch.FloatTensor([5]).to(self.device)
        self.loss_transformer = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.transformer.to(self.device)
        self.meta_opt = torch.optim.Adam(self.transformer.parameters(), lr=1e-5)

        graph_params = []
        graph_params.append({"params": self.gnn.gnn.parameters()})
        graph_params.append({"params": self.gnn.graph_pred_linear.parameters(), "lr":self.learning_rate})
        
        self.opt = optim.Adam(graph_params, lr=self.learning_rate, weight_decay=0) 
        self.gnn.to(torch.device("cuda:0"))
        self.transformer.to(torch.device("cuda:0"))
        
        
        if self.baseline == 0:
            self.ckp_path_gnn = "checkpoints/meta-gtnrp/BIN_PPARA/Meta-GTNRP_GNN_5.pt"
            self.ckp_path_transformer = "checkpoints/meta-gtnrp/BIN_PPARA/Meta-GTNRP_Transformer_5.pt"
                
            self.gnn, self.opt, start_epoch = load_ckp(self.ckp_path_gnn, self.gnn, self.opt)
            self.transformer, self.meta_opt, start_epoch = load_ckp(self.ckp_path_transformer, self.transformer, self.meta_opt)
               
        elif self.baseline == 1:
            self.ckp_path_gnn = "checkpoints/gin/BIN_PPARA/GIN_GNN_5.pt"

            self.gnn, self.opt, start_epoch = load_ckp(self.ckp_path_gnn, self.gnn, self.opt)
         
    def update_graph_params(self, loss, lr_update):
        
        grads = torch.autograd.grad(loss, self.gnn.parameters(), retain_graph=True, allow_unused=True)
        used_grads = [grad for grad in grads if grad is not None]
        
        return parameters_to_vector(used_grads), parameters_to_vector(self.gnn.parameters()) - parameters_to_vector(used_grads) * lr_update
    
    def update_tr_params(self, loss, lr_update):
       
        grads_tr = torch.autograd.grad(loss, self.transformer.parameters())
        used_grads_tr = [grad for grad in grads_tr if grad is not None]
        
        return parameters_to_vector(used_grads_tr), parameters_to_vector(self.transformer.parameters()) - parameters_to_vector(used_grads_tr) * lr_update

    def meta_evaluate(self):

        roc_scores = []
        f1_scores = []
        p_scores = []
        sn_scores = []
        sp_scores = []
        acc_scores = []
        bacc_scores = []

        t=0
        
        graph_params = parameters_to_vector(self.gnn.parameters())
        if self.baseline == 0:
            tr_params = parameters_to_vector(self.transformer.parameters())
            
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        
        for test_task in range(self.test_tasks):
            
            support_set, query_set, smiles_ss, smiles_qs = sample_test(self.tasks, test_task, self.data, self.batch_size, self.n_support, self.n_query)
            self.gnn.eval()
            if self.baseline == 0:
                self.transformer.eval()
                
            for k in range(0, self.k_test):
                
                graph_loss = torch.tensor([0.0]).to(device)
                
                if self.baseline == 0:
                    loss_logits = torch.tensor([0.0]).to(device)
                
                for batch_idx, batch in enumerate(tqdm(support_set, desc="Iteration")):
                    
                    batch = batch.to(device)
                    graph_pred, emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    y = batch.y.view(graph_pred.shape)
                    loss_graph = self.loss(graph_pred.double(), y.to(torch.float64))
                    graph_loss += torch.sum(loss_graph)/graph_pred.size(dim=0)
                    
                    if self.baseline == 0:
                        
                        val_logit, emb = self.transformer(self.gnn.pool(emb, batch.batch))
                        loss_tr = self.loss_transformer(F.sigmoid(val_logit).double(), y.to(torch.float64))
                        loss_logits += torch.sum(loss_tr)/val_logit.size(dim=0)
                          
                    
                updated_grad, updated_params = self.update_graph_params(graph_loss, lr_update = self.lr_update)
                vector_to_parameters(updated_params, self.gnn.parameters())
                
                if self.baseline == 0:
                    updated_grad_tr, updated_tr_params = self.update_tr_params(loss_logits, lr_update = self.lr_update)
                    vector_to_parameters(updated_tr_params, self.transformer.parameters())
            
            nodes=[]
            labels=[]
            y_label = []
            y_pred = []
           
            smiles_list = []
            
            for batch_idx, (batch, batch2) in enumerate(zip(tqdm(query_set, desc="Iteration"), tqdm(smiles_qs, desc="Iteration"))):
                
                batch = batch.to(device)
                
                with torch.no_grad(): 
                    logit, emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    emb = self.gnn.pool(emb, batch.batch)
    
                y_label.append(batch.y.view(logit.shape))
            
                if self.baseline == 0:
                    with torch.no_grad(): 
                        logit, emb = self.transformer(emb)
                
                #print(F.sigmoid(logit))
                  
                pred = parse_pred(logit)
                
                sa_pred = []
                
                for i in pred.cpu().detach().numpy():
                    sa_pred.append(i[0])
                
                #print(batch2)
                
                for i in batch2:
                    smiles_list.append(i)
                
                with open("sa_pred_overall.smi", 'a') as f:
                    f.write("\n".join(str(i) for i in batch2))
                    f.write("\n")
                
                with open("sa_pred_overall.bio", 'a') as f:
                    f.write("\n".join(str(i) for i in sa_pred))
                    f.write("\n")
                
                emb_tsne = emb.cpu().detach().numpy() 
                y_tsne = batch.y.view(pred.shape).cpu().detach().numpy()
               
                for i in emb_tsne:
                    nodes.append(i)
                for j in y_tsne:
                    labels.append(j)
                
                y_pred.append(pred) 
                
            sa = structural_alerts_pred("sa_pred_overall.smi", "sa_pred_overall.bio")
            
            sys.exit("Finish Execution")
            
            roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores  = metrics(roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores, y_label, y_pred)
            
            vector_to_parameters(graph_params, self.gnn.parameters())
            if self.baseline == 0:
                vector_to_parameters(tr_params, self.transformer.parameters())
            
        return [roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores], self.gnn.state_dict(), self.transformer.state_dict(), self.opt.state_dict(), self.meta_opt.state_dict() 