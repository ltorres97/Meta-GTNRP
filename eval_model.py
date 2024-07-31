import torch
from gnntr_eval import GNNTR_eval
import statistics

def save_ckp(state, is_best, checkpoint_dir, filename):
    
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)

def save_result(epoch, N, exp, filename):
            
    file = open(filename, "a")
    
    if epoch < N:
        file.write("Results: " + "\t")
        file.write(str(exp) + "\t")
    if epoch == N:
        file.write("Results: " + "\t")
        file.write(str(exp) + "\n")
        print(str(exp))
        file.write(str(epoch) + " Support Sets: (Mean:"+ str(statistics.mean(exp)) +", SD:" +str(statistics.stdev(exp)) + ") | \t")
    file.write("\n")
    file.close()

dataset = "nura"
gnn = "gin" # gin, graphsage, gcn
support_set = 5 # 5 or 10
pretrained = "pre-trained/supervised_contextpred.pth" # gin: supervised_contextpred.pth, gcn: gcn_supervised_contextpred.pth, graphsage: graphsage_supervised_contextpred.pth
baseline = 0 # 1 for baselines, 0 for Meta-GTNRP
nr = 'PR' # PR, PXR, RXR, GR, AR, ERA, ERB, FXR, PPARD, PPARG, PPARA
task = "bin" # bin, ago, ant
task_name = 'BIN_PR'
model_name = 'Meta-GTNRP' # GIN, GraphSAGE, GCN, Meta-GTNRP
device = "cuda:0"

# Meta-GTNRP - Few-shot GNN-TR architecture
# GIN
# GraphSAGE - Graph Isomorphism Network
# GraphSAGE - Standard Graph Convolutional Network

device = "cuda:0"      
model_eval = GNNTR_eval(dataset, gnn, support_set, pretrained, baseline, nr, task, model_name, task_name)

print("Dataset:", dataset)

roc_auc_list = []

if dataset == "nura":
    roc = []
    f1s = []
    prs = []
    sns = []
    sps = []
    acc = []
    bacc = []
    
N = 30
   
for epoch in range(1, 31):
    
    [roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores], gnn_model, transformer_model, gnn_opt, t_opt = model_eval.meta_evaluate()

    if epoch <= N:
      roc.append(round(roc_scores,4))
      f1s.append(round(f1_scores,4))
      prs.append(round(p_scores,4))
      sns.append(round(sn_scores,4))
      sps.append(round(sp_scores,4))
      acc.append(round(acc_scores,4))
      bacc.append(round(bacc_scores,4))
        
    #save_result(epoch, N, roc, "results-exp/" + task_name + "/mean_" + model_name + "/roc-" + model_name + "_" + str(support_set) + ".txt")
    #save_result(epoch, N, f1s, "results-exp/" + task_name + "/mean_" + model_name + "/f1s-" + model_name + "_" + str(support_set) + ".txt")
    #save_result(epoch, N, prs, "results-exp/" + task_name + "/mean_" + model_name + "/prs-" + model_name + "_" + str(support_set) + ".txt")
    #save_result(epoch, N, sns, "results-exp/" + task_name + "/mean_" + model_name + "/sns-" + model_name + "_" + str(support_set) + ".txt")
    #save_result(epoch, N, sps, "results-exp/" + task_name + "/mean_" + model_name + "/sps-" + model_name + "_" + str(support_set) + ".txt")
    #save_result(epoch, N, acc, "results-exp/" + task_name + "/mean_" + model_name + "/acc-" + model_name + "_" + str(support_set) + ".txt")
    #save_result(epoch, N, bacc, "results-exp/" + task_name + "/mean_" + model_name + "/bacc-" + model_name + "_" + str(support_set) + ".txt")
  