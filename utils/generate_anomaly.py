import json
import random
from itertools import combinations
import torch
import dgl
import numpy as np
from transformers import AutoModel, AutoTokenizer


def add_structure_anomaly(graph, clique_number, clique_size, p=0.2):
    nodes = graph.nodes()
    src = []
    dst = []
    node_list = []
    for i in range(clique_number):
        nodes_choice = random.choices(nodes, k=clique_size)
        node_list.extend(u.item() for u in nodes_choice)
        edges = list(combinations(nodes_choice, 2))
        for u, v in edges:
            if random.random() < p:
                continue
            src.append(u), src.append(v)
            dst.append(v), dst.append(u)
    print("add edges number: ", len(src))
    print("clique node number", len(node_list))
    graph.add_edges(src, dst)
    graph.ndata["label"][node_list] = 1


def add_feature_anomaly(graph, n, text_list):
    model_name = "google-bert/bert-base-uncased"
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    q = int(torch.mean(graph.in_degrees().float()).item() * 2)
    nodes_anomaly = random.choices(graph.nodes(), k=n)
    nodes_anomaly = [node.item() for node in nodes_anomaly]
    graph.ndata["label"][nodes_anomaly] = 1
    for node in nodes_anomaly:
        nodes_choice = random.choices(graph.nodes(), k=q)
        nodes_choice = [i.item() for i in nodes_choice]
        nodes_choice.append(node)
        batch = [text_list[i] for i in nodes_choice]
        tokenize_list = tokenizer(batch, padding=True, truncation=True, return_tensors="pt",
                                  max_length=128)
        tokenize_list = {k: v.to(device) for k, v in tokenize_list.items()}
        with torch.no_grad():
            output = model(**tokenize_list)
        sentence_embedding = output.last_hidden_state[:, 0, :]
        out_node = sentence_embedding[0]
        others = sentence_embedding[1:]
        others_diff = torch.norm(others - out_node, p=2, dim=1)
        out_max = torch.argmax(others_diff)
        text_list[node] = text_list[out_max]
        graph.ndata["feature"][node] = graph.ndata["feature"][out_max]
        torch.cuda.empty_cache()
    return text_list


def main():
    dataset = 'cora'
    if dataset == "cora":
        edge_index = np.load(f"../ori_datasets/{dataset}/{dataset}_edge.npy")
        edge_index = torch.from_numpy(edge_index)
        feature = np.load(f"../ori_datasets/{dataset}/{dataset}_f_bert.npy")
        graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
        tit_dict = json.load(open(f"../ori_datasets/{dataset}/{dataset}_text.json"))
        text_list = []
        for i in range(len(tit_dict)):
            t = str(tit_dict[str(i)])
            text_list.append(t)
        feature = torch.from_numpy(feature)
    elif dataset == "arxiv":
        graph = dgl.load_graphs(f"../ori_datasets/{dataset}/dgl_data_processed")[0][0]
        f = open(f"../ori_datasets/{dataset}/text.txt", encoding="utf-8")
        text_list = f.readlines()
        feature = graph.ndata["feat"]
    elif dataset == "cora":
        graph = dgl.load_graphs(f"../ori_datasets/cora/cora")[0][0]
        f = open(f"../ori_datasets/{dataset}/text.txt", encoding="utf-8")
        text_list = f.readlines()
        feature = graph.ndata["feature"]

    graph = dgl.to_bidirected(graph)
    graph: dgl.DGLGraph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    graph.ndata["feature"] = feature.float()
    node_num = graph.nodes().shape[0]

    label = torch.zeros(node_num)
    graph.ndata["label"] = label
    print(f"node number: {node_num}, original edges: {graph.edges()[0].shape[0]}", )
    n = int(graph.number_of_nodes() * 0.025)
    print("n:", n)
    degrees = graph.in_degrees().float()
    avg_degrees = torch.mean(degrees).item()
    m = int(2 * avg_degrees)
    print("m: ", m)
    clique_number = int(n / m)
    text_list = add_feature_anomaly(graph, n, text_list)
    print("labels: ", sum(graph.ndata["label"]).item())
    add_structure_anomaly(graph, clique_number, m, p=0)
    print("edges: ", graph.edges()[0].shape[0])
    print("labels: ", sum(graph.ndata["label"]).item())
    dgl.save_graphs(f"./{dataset}/{dataset}", [graph])
    f = open(f"./{dataset}/text.txt", "w+")
    for text in text_list:
        text_a = text + "\n"
        f.write(text_a)
    f.close()


if __name__ == "__main__":
    main()
