import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from pygod.metric import eval_recall_at_k
import torch



def eval_model(labels: np.ndarray, pred: np.ndarray, time: str = "", file_name: str = "", save=True):
    auc = roc_auc_score(labels, pred)
    recall_k = eval_recall_at_k(torch.Tensor(labels), torch.Tensor(pred), np.sum(labels == 1).item())
    pred = pred.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    f1_macro = f1_score(labels, pred, average='macro')
    if save or file_name != "":
        file_dir = f'../results/{file_name}.txt'
        f = open(file_dir, 'a+')
        f.write(
            f"AUROC: {auc}\tF1-Macro: {f1_macro}\trecall@k: {recall_k}\ttime:{time}\n")
        f.close()
        print(f'save to file name: {file_name}')    
    # print(nodes[anomaly_true])
    return auc, f1_macro, recall_k
