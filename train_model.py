import torch
from gnntr_train import GNNTR

def save_ckp(state, checkpoint_dir, filename, best):
    
    if best == True:
        f_path = checkpoint_dir + filename
        torch.save(state, f_path)

dataset = "nura"
gnn = "gin" #gin, graphsage, gcn
support_set = 10
pretrained = "pre-trained/supervised_contextpred.pth"
baseline = 0
device = "cuda:0"
model = GNNTR(dataset, gnn, support_set, pretrained, baseline)

if dataset == "nura":
    roc = [0]
    f1s = [0]
    prs = [0]
    sns = [0]
    sps = [0]
    acc = [0]
    bacc = [0]

    labels =  ['BIN_PR']
    
for epoch in range(1, 2001):
    
    best = False

    model.meta_train()
    
    [roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores], gnn_model, transformer_model, gnn_opt, t_opt = model.meta_test() 
    
    print(roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores)

    checkpoint_gnn = {
            'epoch': epoch + 1,
            'state_dict': gnn_model,
            'optimizer': gnn_opt
    }
    
    checkpoint_transformer = {
            'epoch': epoch + 1,
            'state_dict': transformer_model,
            'optimizer': t_opt
    }
    
    checkpoint_dir = 'checkpoints/meta-gtnrp/BIN_PR/'

    for i in range(0, len(roc_scores)):
        if roc[i] < roc_scores[i]:
            roc[i] = roc_scores[i]
            best = True
    
    for i in range(0, len(f1_scores)):
        if f1s[i] < f1_scores[i]:
            f1s[i] = f1_scores[i]
    
    for i in range(0, len(p_scores)):
        if prs[i] < p_scores[i]:
            prs[i] = p_scores[i]

    for i in range(0, len(sn_scores)):
        if sns[i] < sn_scores[i]:
            sns[i] = sn_scores[i]
    
    for i in range(0, len(sp_scores)):
        if sps[i] < sp_scores[i]:
            sps[i] = sp_scores[i]

    for i in range(0, len(acc_scores)):
        if acc[i] < acc_scores[i]:
            acc[i] = acc_scores[i]
    
    for i in range(0, len(bacc_scores)):
        if bacc[i] < bacc_scores[i]:
            bacc[i] = bacc_scores[i]
    
    filename = "results-exp/BIN_PR/Meta-GTNRP-10.txt"
    file = open(filename, "a")
    file.write("[ROC-AUC, F1-score, Precision, Sensitivity, Specificity, Accuracy, Balanced Accuracy]:\t")
    file.write("[" + str(roc_scores) + "," + str(f1_scores) + "," + str(p_scores) + "," + str(sn_scores) + "," + str(sp_scores) + "," + str(acc_scores) + "," + str(bacc_scores) + "]" + "\t ; ")
    file.write("[" + str(roc) + "," +  str(f1s) + "," + str(prs) + "," + str(sns) + "," + str(sps) + "," + str(acc) + "," + str(bacc) + "]") 
    file.write("\n")
    file.close()
    
    if baseline == 0:
      save_ckp(checkpoint_gnn, checkpoint_dir, "Meta-GTNRP_GNN_10.pt", best)
      save_ckp(checkpoint_transformer, checkpoint_dir, "Meta-GTNRP_Transformer_10.pt", best)

    if baseline == 1:
       save_ckp(checkpoint_gnn, checkpoint_dir, "GIN_GNN_10.pt", best)
        
   
