import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from layers import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor
from torch_geometric.utils import dropout_adj
from model import Encoder, Model, drop_feature
from utils import normalize_adj_tensor, normalize_adj_tensor_sp, edge2adj
from attack import PGD_attack_graph
from eval import label_classification

def train(model: Model, x, edge_index, eps, lamb, alpha, beta, steps, node_ratio):
    optimizer.zero_grad()
    adj = edge2adj(x, edge_index)
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]

    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)  

    adj_1 = edge2adj(x_1, edge_index_1)
    adj_2 = edge2adj(x_2, edge_index_2)
    
    if eps > 0:
        adj_3, x_3 = PGD_attack_graph(model, edge_index_1, edge_index, x_1, x, steps, node_ratio, alpha, beta)
    z = model(x, adj)
    z_1 = model(x_1, adj_1)
    z_2 = model(x_2, adj_2)
    loss1, simi1 = model.loss(z_1,z_2,batch_size=0)
    loss2, simi2 = model.loss(z_1,z,batch_size=0)
    loss3, simi3 = model.loss(z_2,z,batch_size=0)
    loss1 = loss1.mean() + lamb*torch.clamp(simi1*2 - simi2.detach()-simi3.detach(), 0).mean()
    if eps > 0:
        z_3 = model(x_3,adj_3)
        loss2, _ = model.loss(z_1,z_3)
        loss2 = loss2.mean()
        loss = (loss1 + eps*loss2)
    else: 
        loss = loss1
        loss2 = loss1

    loss.backward()
    optimizer.step()

    return loss1.item(), loss2.item()

def test(model: Model, x, edge_index, y, final=False, task ="node"):   
    model.eval()
    adj = edge2adj(x, edge_index)
    x = x.to(device)
    adj = adj.to(device)
    z = model(x, normalize_adj_tensor_sp(adj))
    return label_classification(z, y, ratio=0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--log', type=str, default='results/tmp')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--lamb', type=float, default=0)
    args = parser.parse_args()


    assert args.gpu_id in range(0, 8)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config["seed"])
    random.seed(12345)
    np.random.seed(config["seed"])
    
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU()})[config['activation']]
    base_model = GCNConv
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    eps = config["eps"]
    lamb = config["lamb"]

    sample_size = 500
    
    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', "AmazonC", "AmazonP", 'CoauthorC', 'CoauthorP']
        if name == "AmazonC":
            return Amazon(path, "Computers", T.NormalizeFeatures())
        if name == "AmazonP":
            return Amazon(path, "Photo", T.NormalizeFeatures())
        if name == 'CoauthorC':
            return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
        if name == 'CoauthorP':
            return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

        return Planetoid(
            path,
            name,
            "public",
            T.NormalizeFeatures())
        
    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset.data  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(data.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start    
    G = nx.Graph()
    G.add_edges_from(list(zip(data.edge_index.numpy()[0],data.edge_index.numpy()[1])))
    
    model.train()
    for epoch in range(1, num_epochs + 1):
         # increase the eps every T epochs
        if epoch%20 ==0:
            eps = eps*1.1
        # sample a subgraph from the original one
        S = G.subgraph(np.random.permutation(G.number_of_nodes())[:sample_size])
        x = data.x[np.array(S.nodes())].to(device)
        S = nx.relabel.convert_node_labels_to_integers(S, first_label=0, ordering='default')
        edge_index = np.array(S.edges()).T
        edge_index = torch.LongTensor(np.hstack([edge_index,edge_index[::-1]])).to(device)
        loss1, loss2 = train(model, x, edge_index, eps, lamb, config["alpha"], config["beta"], 5, 0.2)
            
        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss1={loss1:.4f}, loss2={loss2:.4f}'
              f' this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    results = test(model, data.x, data.edge_index, data.y, final=True)
    print(results)
    with open(osp.join(args.log, "progress.csv"), "w") as f:
        f.write(str(results))
   