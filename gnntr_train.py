import torch
import torch.nn as nn
from gnn_tr import GNN_prediction, TR
import torch.nn.functional as F
from data import MoleculeDataset, random_sampler
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, balanced_accuracy_score
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

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
    print(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
      
    optimizer_to(optimizer, device)

    return model, optimizer, checkpoint['epoch']


def sample_train(data, batch_size, n_support, n_query, nr, task):
    
    support_sets = []
    
    query_sets = []
    
    T = {'PR': 1, 'PXR': 2, 'RXR': 3, 'GR': 4, 'AR': 5, 'ERA': 6, 'ERB': 7, 'FXR': 8, 'PPARD': 9, 'PPARG': 10, 'PPARA': 11}
    
    train_tasks = [1,2,3,4,5,6,7,8,9,10,11] # replace with all tasks
        
    train_tasks.remove(T[nr]) # remove the task for meta-testing to set the meta-training tasks
    
    if task == "ago" or task == "ant":
        train_tasks = [T[nr]] # single-task model
            
    for train_task in train_tasks:
         dataset = MoleculeDataset("Data/" + data + "/pre-processed_BIN/task_" + str(train_task), dataset = data)
         support_dataset, query_dataset = random_sampler(dataset, data, train_task-1, n_support, n_query, train=True, task="bin")
         support_set = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last = True)
         query_set = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last = True)
         support_sets.append(support_set)
         query_sets.append(query_set)
         
    return support_sets, query_sets

def sample_test(data, batch_size, n_support, n_query, nr, task):
    
    T = {'PR': 1, 'PXR': 2, 'RXR': 3, 'GR': 4, 'AR': 5, 'ERA': 6, 'ERB': 7, 'FXR': 8, 'PPARD': 9, 'PPARG': 10, 'PPARA': 11}
    
    test_task = T[nr] # replace with test task
    
    train=False
    
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
   
    roc_auc_list.append(roc_auc_score(y_label, y_pred))
    roc_auc = sum(roc_auc_list)/len(roc_auc_list)

    f1_score_list.append(f1_score(y_label, y_pred, average='weighted'))
    f1_scr = sum(f1_score_list)/len(f1_score_list)

    precision_score_list.append(precision_score(y_label, y_pred, average='weighted'))
    p_scr = sum(precision_score_list)/len(precision_score_list)

    sn_score_list.append(recall_score(y_label, y_pred, average='weighted'))
    sn_scr = sum(sn_score_list)/len(sn_score_list)

    tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
    sp_score_list.append(tn/(tn+fp))
    sp_scr = sum(sp_score_list)/len(sp_score_list)

    acc_score_list.append(accuracy_score(y_label, y_pred))
    acc_scr =  sum(acc_score_list)/len(acc_score_list)

    bacc_score_list.append(balanced_accuracy_score(y_label, y_pred))
    bacc_scr =  sum(bacc_score_list)/len(bacc_score_list)

    roc_scores.append(roc_auc)
    f1_scores.append(f1_scr)
    p_scores.append(p_scr)
    sn_scores.append(sn_scr)
    sp_scores.append(sp_scr)
    acc_scores.append(acc_scr)
    bacc_scores.append(bacc_scr)

    return roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores

def parse_pred(logit):
    
    pred = F.sigmoid(logit)
    pred = torch.where(pred>0.5, torch.ones_like(pred), pred)
    pred = torch.where(pred<=0.5, torch.zeros_like(pred), pred) 
    
    return pred

class GNNTR():
    def __init__(self, dataset, gnn, support_set, pretrained, baseline, nr, task):
        super(GNNTR,self).__init__()
        
        if dataset == "nura":
            
            if task =="bin":
                self.tasks = 11
                self.train_tasks  = 10
                self.test_tasks = 1
                
            if task == "ago" or task == "ant":
                self.tasks = 2
                self.train_tasks  = 1
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
            
            self.ckp_path_gnn = "checkpoints/meta-gtnrp/BIN_PR/Meta-GTNRP_GNN_10.pt"
            self.ckp_path_transformer = "checkpoints/meta-gtnrp/BIN_PR/Meta-GTNRP_Transformer_10.pt"
            
            #self.gnn, self.opt, start_epoch = load_ckp(self.ckp_path_gnn, self.gnn, self.opt)
            #self.transformer, self.meta_opt, start_epoch = load_ckp(self.ckp_path_transformer, self.transformer, self.meta_opt)
               
        elif self.baseline == 1:
            
            self.ckp_path_gnn = "checkpoints/gin/BIN_PR/GIN_GNN_10.pt"
            #self.gnn, self.opt, start_epoch = load_ckp(self.ckp_path_gnn, self.gnn, self.opt)
      
    def update_graph_params(self, loss, lr_update):
        
        grads = torch.autograd.grad(loss, self.gnn.parameters(), retain_graph=True, allow_unused=True)
        used_grads = [grad for grad in grads if grad is not None]
        
        return parameters_to_vector(used_grads), parameters_to_vector(self.gnn.parameters()) - parameters_to_vector(used_grads) * lr_update
    
    def update_tr_params(self, loss, lr_update):
       
        grads_tr = torch.autograd.grad(loss, self.transformer.parameters())
        used_grads_tr = [grad for grad in grads_tr if grad is not None]
        
        return parameters_to_vector(used_grads_tr), parameters_to_vector(self.transformer.parameters()) - parameters_to_vector(used_grads_tr) * lr_update

    def meta_train(self):
        self.gnn.train()
        if self.baseline == 0:
            self.transformer.train()
        
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
                
        support_sets, query_sets = sample_train(self.data, self.batch_size, self.n_support, self.n_query, self.nr, self.task)
        
        for k in range(0, self.k_train):
            graph_params = parameters_to_vector(self.gnn.parameters())
            if self.baseline == 0:
                tr_params = parameters_to_vector(self.transformer.parameters())
            query_losses = torch.tensor([0.0]).to(device)
            for t in range(self.train_tasks):
                
                loss_support = torch.tensor([0.0]).to(device)
                loss_query = torch.tensor([0.0]).to(device)

                if self.baseline == 0:
                    inner_losses = torch.tensor([0.0]).to(device)
                    outer_losses = torch.tensor([0.0]).to(device)
                    
                for batch_idx, batch in enumerate(tqdm(support_sets[t], desc="Iteration")):
                    batch = batch.to(device)
                    graph_pred, graph_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    #print(batch.y)
                    label = batch.y.view(graph_pred.shape)
                    #print(label)
                    loss_graph = self.loss(graph_pred.double(), label.to(torch.float64))
                    support_loss = torch.sum(loss_graph)/graph_pred.size(dim=0)
                    loss_support += support_loss

                    if self.baseline == 0:
                        pred, emb_tr = self.transformer(self.gnn.pool(graph_emb, batch.batch))
                        loss_tr = self.loss_transformer(F.sigmoid(pred).double(), label.to(torch.float64)) 
                        inner_loss = torch.sum(loss_tr)/pred.size(dim=0)
                   
                    if self.baseline == 0:
                        inner_losses += inner_loss
                
                updated_grad, updated_params = self.update_graph_params(loss_support, lr_update = self.lr_update)
                vector_to_parameters(updated_params, self.gnn.parameters())

                if self.baseline == 0:
                    updated_grad_tr, updated_tr_params = self.update_tr_params(inner_losses, lr_update = self.lr_update)
                    vector_to_parameters(updated_tr_params, self.transformer.parameters())
                
                for batch_idx, batch in enumerate(tqdm(query_sets[t], desc="Iteration")):
                    batch = batch.to(device)
                    
                    graph_pred, emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    label = batch.y.view(graph_pred.shape)
                    loss_graph = self.loss(graph_pred.double(), label.to(torch.float64))
                    query_loss = torch.sum(loss_graph)/graph_pred.size(dim=0)
                    loss_query += query_loss
                    
                    if self.baseline == 0:
                        logit, emb = self.transformer(self.gnn.pool(emb, batch.batch))
                        loss_tr = self.loss_transformer(F.sigmoid(logit).double(), label.to(torch.float64))
                        outer_loss = torch.sum(loss_tr)/logit.size(dim=0) 
                                        
                    if self.baseline == 0:
                        outer_losses += outer_loss
              
                if t == 0:
                    query_losses = loss_query
                    if self.baseline == 0:
                        query_outer_losses = outer_losses
                else:
                    query_losses = torch.cat((query_losses, loss_query), 0)
                    if self.baseline == 0:
                        query_outer_losses =  torch.cat((query_outer_losses, outer_losses), 0)

                vector_to_parameters(graph_params, self.gnn.parameters())
                if self.baseline == 0:
                    vector_to_parameters(tr_params, self.transformer.parameters())
            
            query_losses = torch.sum(query_losses)
                
            loss_graph = query_losses / self.train_tasks   
            loss_graph.to(device)
            
            if self.baseline == 0:
                query_outer_loss = torch.sum(query_outer_losses)
                loss_tr = query_outer_loss / self.train_tasks
                loss_tr.to(device) 
            
            self.opt.zero_grad()
            
            if self.baseline == 0:
                self.meta_opt.zero_grad()
            
            if self.baseline == 1:
                loss_graph.backward()
            
            if self.baseline == 0:
                loss_graph.backward(retain_graph=True)
                loss_tr.backward()
            
            self.opt.step()
            
            if self.baseline == 0:
                self.meta_opt.step()
            
        return []

    def meta_test(self):

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
    
                y_label.append(batch.y.view(logit.shape))
            
                if self.baseline == 0:
                    with torch.no_grad(): 
                        logit, emb = self.transformer(self.gnn.pool(emb, batch.batch))
                
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