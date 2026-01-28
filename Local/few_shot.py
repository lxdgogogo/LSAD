import argparse
import os
import random
import sys
import time
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import datetime

os.chdir(sys.path[0])
sys.path.append("..")

from utils.early_stop import EarlyStopping
from utils.load_data import load_data
from utils.eval_utils import eval_model
from Global.anomaly_clip import AnomalyClip
from utils.dataset import train_test
from utils.functions import print_trainable_parameters
from PG_few import PG_few
# from PG_few_dynamic import PG_few



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    epoch_num = args.epoch  # 200
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    dataset = args.dataset  # cora pubmed arxiv
    dropout = args.dropout
    batch_size = args.batch_size  # 64
    alpha = args.alpha
    norm = "batchnorm"
    activation = "prelu"
    epsilon = 0.98 # 0.98
    lamb = args.lamb  # 0.5
    random.seed(42)
    start = time.time()
    shot_num = args.shot_num
    device = args.device
    file_name = f"{dataset}_few_{shot_num}"
    random.seed(42)

    graph = load_data(dataset, device).int()
    label = graph.ndata["label"]
    print("node sum: ", graph.number_of_nodes())
    train_idx, train_label, test_idx, test_label = train_test(label, few_shot_num=shot_num, device=device)
    feature = graph.ndata["feature"]
    text_emb = graph.ndata['text_embedding'].to(device)
    last_hidden_size = text_emb.shape[1]
    pretrain = AnomalyClip(feature.shape[1], last_hidden_size, feature.shape[1], num_layers, dropout, norm, activation,
                            alpha, device).to(device)
    now_date = datetime.date.today().strftime("%Y-%m-%d")
    pretrain.load_state_dict(
        torch.load(f'../model/{dataset}_{now_date}.pth'))
    pretrain.eval()
    with torch.no_grad():
        emb = pretrain.text_projection(text_emb)
        encode_feature = pretrain.encode_gnn(graph, feature)
        pred, _ = pretrain(graph, text_emb, feature, pre=False, batch_size=batch_size)
    eval_model(test_label.cpu().numpy(), pred[test_idx].cpu().numpy(), save=False)
    pg = PG_few(encode_feature, emb, epsilon, lamb, alpha, device).to(device)
    print_trainable_parameters(pg)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pg.parameters()), lr=lr, weight_decay=weight_decay)
    epoch_iter = tqdm(range(epoch_num))
    for epoch in epoch_iter:
        score, loss = pg(graph, train_idx, train_label)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        auc = roc_auc_score(train_label.detach().cpu().numpy(), score.detach().cpu().numpy())
        epoch_iter.set_description(f'Epoch {epoch}: train_loss: {loss} auc: {auc}')

    pg.eval()
    with torch.no_grad():
        score = pg.inference(graph, graph.nodes())
    score = (1 - lamb) * pred + lamb * score
    score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
    end = time.time()
    eval_model(test_label.cpu().numpy(), score[test_idx].detach().cpu().numpy(), str(end - start), file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')  
    parser.add_argument('--shot_num', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=20)  # 300
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lamb', type=float, default=0.5)  # 0.5
    parser.add_argument('--alpha', type=float, default=0.8)  # 0.3
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    main(args)
