import torch
import torch.nn as nn
from gnn_tr import GNN_prediction, TR
import torch.nn.functional as F
from data import MoleculeDataset, random_sampler
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, balanced_accuracy_score
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from sklearn.manifold import TSNE
#from tsnecuda import TSNE # Use this package if the previous one doesn't work
import matplotlib.pyplot as plt
import pandas as pd

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

def sample_test(data, batch_size, n_support, n_query, nr, task):
    
    T = {'PR': 1, 'PXR': 2, 'RXR': 3, 'GR': 4, 'AR': 5, 'ERA': 6, 'ERB': 7, 'FXR': 8, 'PPARD': 9, 'PPARG': 10, 'PPARA': 11}
    
    test_task = T[nr] # replace with test task
    
    train = False
    
    print(test_task)
    
    dataset = MoleculeDataset("Data/" + data + "/pre-processed_" + task.upper() + "/task_" + str(test_task), dataset = data)
    support_dataset, query_dataset = random_sampler(dataset, data, test_task-1, n_support, n_query, train, task)
    support_set = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=True)
    query_set = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last = True)
    
    return support_set, query_set
    

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


def plot_tsne(nodes, labels, t):
    
    #Plot t-SNE visualizations
    
    title = 'Meta-GTNRP'
    labels_list = ['PR']
    t+=1
        
    emb_tsne = np.asarray(nodes)
    y_tsne = np.asarray(labels).flatten()
    
    print(emb_tsne.shape)
    print(y_tsne.shape)
      
    z = TSNE(n_components=2, init='random').fit_transform(emb_tsne)
    label_vals = {0: 'Negative', 1: 'Positive'}
    print(y_tsne.size)
    print(z.size)
    tsne_result_df = pd.DataFrame({'tsne_dim_1': z[:,0], 'tsne_dim_2': z[:,1], 'label': y_tsne})
    tsne_result_df['label'] = tsne_result_df['label'].map(label_vals)
    fig, ax = plt.subplots(1)
    
    m_colors = ["blue", "red"]
    for (lab, group), col in zip(tsne_result_df.groupby("label"), m_colors):
       ax.scatter(group.tsne_dim_1, group.tsne_dim_2, edgecolors=col, facecolors="white", alpha = 1, s = 5, linewidth = 0.4, label=lab)
    
    lim = (z.min()-5, z.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal') 
    #fig.set_figwidth(6.5)
    #fig.set_figheight(3.5)
    handles, labels = ax.get_legend_handles_labels()
    if t == 1:
        ax.legend(handles[:2], labels[:2], bbox_to_anchor=(1, 0.02), loc='lower right')
    #ax.set_title(title)
    ax.set(xlabel="Dimension 1")
    ax.set(ylabel="Dimension 2")
    ax.tick_params(bottom=True) 
    ax.tick_params(left=True)
    plt.grid(False)
    plt.savefig('plots/'+title+"-"+labels_list[t-1]+'-10shot', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return t


class GNNTR_eval():
    def __init__(self, dataset, gnn, support_set, pretrained, baseline, nr, task, model_name, task_name):
        super(GNNTR_eval,self).__init__()
                
        if dataset == "nura":
            self.tasks = 11
            self.train_tasks  = 10
            self.test_tasks = 1
            
        self.data = dataset
        self.baseline = baseline
        self.nr = nr
        self.task = task
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
        self.gnn = GNN_prediction(self.graph_layers, self.emb_size, jk = "last", dropout_prob = 0.5, pooling = "mean", gnn_type = gnn)
        self.transformer = TR(300, (30,1), 1, 128, 5, 5, 256) 
        self.gnn.from_pretrained(pretrained)
        if self.baseline == 0:
            w = 5
        elif self.baseline == 1:
            w = 1
        self.pos_weight = torch.FloatTensor([w]).to(self.device)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.loss_transformer = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.meta_opt = torch.optim.Adam(self.transformer.parameters(), lr=1e-5)
        
        graph_params = []
        graph_params.append({"params": self.gnn.gnn.parameters()})
        graph_params.append({"params": self.gnn.graph_pred_linear.parameters(), "lr":self.learning_rate})
        
        self.opt = optim.Adam(graph_params, lr=self.learning_rate, weight_decay=0) 
        self.gnn.to(torch.device("cuda:0"))
        self.transformer.to(torch.device("cuda:0"))
        
        if self.baseline == 0:
            
            self.ckp_path_gnn = "checkpoints/meta-gtnrp/" + task_name + "/" + model_name + "_GNN_" + str(support_set) + ".pt"
            self.ckp_path_transformer = "checkpoints/meta-gtnrp/" + task_name + "/" + model_name + "_Transformer_" + str(support_set) + ".pt"
            
            self.gnn, self.opt, start_epoch = load_ckp(self.ckp_path_gnn, self.gnn, self.opt)
            self.transformer, self.meta_opt, start_epoch = load_ckp(self.ckp_path_transformer, self.transformer, self.meta_opt)
               
        elif self.baseline == 1:

            self.ckp_path_gnn = "checkpoints/" + gnn + "/" + task_name + "/" + model_name + "_GNN_" + str(support_set) + ".pt"
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
            
            support_set, query_set = sample_test(self.data, self.batch_size, self.n_support, self.n_query, self.nr, self.task)
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
           
            for batch_idx, batch in enumerate(tqdm(query_set, desc="Iteration")):
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
                
                emb_tsne = emb.cpu().detach().numpy() 
                y_tsne = batch.y.view(pred.shape).cpu().detach().numpy()
               
                for i in emb_tsne:
                    nodes.append(i)
                for j in y_tsne:
                    labels.append(j)
                
                y_pred.append(pred)   
                
            #t = plot_tsne(nodes, labels, t)
             
            roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores  = metrics(roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores, y_label, y_pred)
            
            vector_to_parameters(graph_params, self.gnn.parameters())
            if self.baseline == 0:
                vector_to_parameters(tr_params, self.transformer.parameters())
            
        return [roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores], self.gnn.state_dict(), self.transformer.state_dict(), self.opt.state_dict(), self.meta_opt.state_dict() 