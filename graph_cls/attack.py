import torch
import numpy as np
from utils import edge2adj

def bisection(a,eps,xi,ub=1):
    pa = torch.clamp(a, 0, ub)
    if torch.sum(pa) <= eps:
        upper_S_update = pa
    else:
        mu_l = torch.min(a-1)
        mu_u = torch.max(a)
        mu_a = (mu_u + mu_l)/2
        while torch.abs(mu_u - mu_l)>xi:
            mu_a = (mu_u + mu_l)/2
            gu = torch.sum(torch.clamp(a-mu_a, 0, ub)) - eps
            gu_l = torch.sum(torch.clamp(a-mu_l, 0, ub)) - eps
            if gu == 0:
                break
            if torch.sign(gu) == torch.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a
        upper_S_update = torch.clamp(a-mu_a, 0, ub)
    return upper_S_update


def PGD_attack_graph(model, data, data_aug, steps, node_ratio, alpha, beta):
    """ PGD attack on both features and edges"""
    for param in  model.parameters():
        param.requires_grad = False
        
    x_1, edge_index_1 = data_aug.x, data_aug.edge_index
    x_2, edge_index_2 = data.x, data.edge_index
    model.eval()
    device = x_1.device
    total_edges = edge_index_2.shape[1]
    n_node = x_2.shape[0]
    eps = total_edges * node_ratio/2
    xi = 1e-3
    
    A_ = torch.sparse.FloatTensor(edge_index_2, torch.ones(total_edges,device=device), torch.Size((n_node, n_node))).to_dense() 
    C_ = torch.ones_like(A_) - 2 * A_ #- torch.eye(A_.shape[0],device=device)
    S_ = torch.zeros_like(A_, requires_grad= True)
    _, counts = torch.unique(data.batch, return_counts=True)
    mask = torch.block_diag(*[torch.tril(torch.ones(c,c),diagonal=-1) for c in counts]).to(device)
    delta = torch.zeros_like(x_2, device=device, requires_grad=True)
    adj_1 = edge2adj(x_1, edge_index_1)
    model.to(device)
    for epoch in range(steps):
        S = (S_ * mask)
        S = S + S.T
        A_prime = A_ + (S * C_)
        z1 = model(x_1, adj_1, data_aug.batch, data_aug.num_graphs)
        z2 = model(x_2 + delta, A_prime, data.batch, data.num_graphs) 
        loss = model.loss_cal(z1, z2) 
        loss.backward()
        S_.data = (S_.data + alpha/np.sqrt(epoch+1)*S_.grad.detach()) # annealing
        S_.data = bisection(S_.data, eps, xi) # clip S
        S_.grad.zero_()
        
        delta.data = (delta.data + beta*delta.grad.detach().sign()).clamp(-0.04,0.04)        
        delta.grad.zero_()
    
    randm = torch.rand(n_node, n_node,device=device)
    discretized_S = torch.where(S_.detach() > randm, torch.ones(n_node, n_node,device=device), torch.zeros(n_node, n_node, device=device))
    discretized_S = discretized_S + discretized_S.T
    A_hat = A_ + discretized_S * C_
        
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    x_hat = x_2 + delta.data.to(device)
    assert torch.equal(A_hat, A_hat.transpose(0,1))
    return A_hat, x_hat