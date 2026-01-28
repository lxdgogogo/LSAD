import argparse
import os
import sys
import time
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
import yaml
import random

os.chdir(sys.path[0])
sys.path.append("..")
from utils.early_stop import EarlyStopping
from utils.functions import print_trainable_parameters
from utils.eval_utils import eval_model
from utils.load_data import load_data
from Global.anomaly_clip import AnomalyClip
from utils.dataset import train_test
from Local.PG_few import PG_few


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"Seed set to: {seed}")


def main(args):
    f = open("./Config/config.yaml", "r")
    dataset = args.dataset
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    # alpha = config["alpha"][dataset]
    # lamb = config["lamb"][dataset]
    # max_epoch = config["epoch"][dataset]

    alpha = config["alpha"]['TWIBOT_text']
    lamb = config["lamb"]['TWIBOT_text']
    max_epoch = config["epoch"]['TWIBOT_text']

    batch_size = args.batch_size  # 64 400
    patience = args.patience
    dropout = args.dropout
    num_layers = args.num_layers
    norm = "batchnorm"
    activation = "prelu"
    file_name = args.file_name
    hidden_size = args.hidden_size
    shot_num = args.shot_num
    epoch_few_shot = args.epoch_few_shot
    lr = args.lr
    weight_decay = args.weight_decay
    if file_name == "":
        file_name = f"{dataset}_all"
    device = f"cuda:{args.device}"

    seed = int(time.time())
    set_seed(seed)
    # dataset = "arxiv"
    graph = load_data(dataset, device)
    label = graph.ndata['label']
    feature = graph.ndata["feature"].float()
    text_embeddings = graph.ndata["text_embedding"].to(device)
    print("text embeddings shape: ", text_embeddings.shape)
    last_hidden_size = text_embeddings.shape[1]
    print(last_hidden_size)
    trials = 1
    auc_list = []
    f1_macro_list = []
    recall_list = []
    g_mean_list = []
    time_list = []
    for _ in range(trials):
        train_idx, train_label, test_idx, test_label = train_test(label, few_shot_num=shot_num, device=device)
        stopper = EarlyStopping(patience=patience, dataset=f"{dataset}", save=False)
        pretrain = AnomalyClip(feature.shape[1], last_hidden_size, hidden_size, num_layers, dropout, norm, activation,
                           alpha, batch_size, device).to(device)
        epoch_iter = tqdm(range(max_epoch))
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, pretrain.parameters()), lr, weight_decay=weight_decay)
        print_trainable_parameters(pretrain)
        pre = True
        start = time.time()
        for epoch in epoch_iter:
            if epoch > (max_epoch / 5):
                pre = False  # True when for yelpzip
            score, loss = pretrain(graph, text_embeddings, feature, pre)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            auc = roc_auc_score(label.cpu().numpy(), score.detach().cpu().numpy())
            epoch_iter.set_description(f'Epoch {epoch}: train_loss: {loss}, auc: {auc}')
            print(f'Epoch {epoch}: train_loss: {loss}, auc: {auc}')
            if pre == False:
                stopper.step(loss, pretrain)
                if stopper.early_stop:
                    break
        print(f'Epoch {epoch}: train_loss mean: {loss/graph.num_nodes()}')
        end = time.time()
        pretrain.eval()
        with torch.no_grad():
            pred, loss = pretrain(graph, text_embeddings, feature, pre)
            encode_feature = pretrain.encode_gnn(graph, feature)
            emb_proj = pretrain.text_projection(text_embeddings)
        auc, f1_macro, recall, g_mean = eval_model(label.cpu().numpy(), pred.detach().cpu().numpy(), save=False)
        print(f"Pretrain AUROC: {auc}")
        # pg = PG_few(feature, feature, lamb, alpha, device).to(device)
        pg = PG_few(emb_proj, lamb, K=8, device=device).to(device)
        # pg = PG_few_regular(encode_feature, emb_proj, lamb, alpha, device).to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pg.parameters()), lr=lr, weight_decay=weight_decay)
        epoch_iter = tqdm(range(epoch_few_shot))
        for epoch in epoch_iter:
            score, loss = pg(graph, train_idx, train_label)
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            auc = roc_auc_score(train_label.cpu().numpy(), score.detach().cpu().numpy())
            epoch_iter.set_description(f'Epoch {epoch}: train_loss: {loss} auc: {auc}')
        pg.eval()
        with torch.no_grad():
            score = pg.inference(graph, graph.nodes())
        final_score = (1 - lamb) * pred + lamb * score
        end = time.time()
        auc, f1_macro, recall_k, g_mean = eval_model(test_label.cpu().numpy(), final_score[test_idx].detach().cpu().numpy(), False)
        auc_list.append(auc)
        f1_macro_list.append(f1_macro)
        recall_list.append(recall_k)
        g_mean_list.append(g_mean)
        time_list.append(end - start)
    f = open(f"./results/{dataset}_all.txt", "a+")
    f.write(f"Shot num: {shot_num}, AUC: %.4f ± %.4f, F1-Macro: %.4f ± %.4f, Recall: %.4f ± %.4f, G-mean: %.4f ± %.4f, Time: {sum(time_list)/len(time_list)}\n" % 
    (np.mean(auc_list), np.std(auc_list), np.mean(f1_macro_list), np.std(f1_macro_list), np.mean(recall_list), np.std(recall_list), 
    np.mean(g_mean_list), np.std(g_mean_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora_min_text')  # cora_min_text arxiv_new_text pubmed_text yelpzip_text TWIBOT_text cora_min_v1_text
    parser.add_argument('--epoch_few_shot', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hidden_size', type=int, default=1024)  # 1024
    parser.add_argument('--file_name', type=str, default="")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--shot_num', type=int, default=2)
    parser.add_argument('--alpha', type=int, default=0.1)
    args = parser.parse_args()
    # alpha_list = [1.0]
    # for alpha in alpha_list:
    #     args.alpha = alpha
    #     main(args)
    main(args)
