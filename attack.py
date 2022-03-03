import torch
from utils import normalize_adj_tensor, normalize_adj_tensor_sp, edge2adj

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


def PGD_attack_graph(model, edge_index_1, edge_index_2, x_1, x_2, steps, node_ratio, alpha, beta):
    """ PGD attack on both features and edges"""
    for param in  model.parameters():
        param.requires_grad = False
    model.eval()
    device = x_1.device
    total_edges = edge_index_2.shape[1]
    n_node = x_2.shape[0]
    eps = total_edges * node_ratio/2
    xi = 1e-5
    
    A_ = torch.sparse.FloatTensor(edge_index_2, torch.ones(total_edges,device=device), torch.Size((n_node, n_node))).to_dense() 
    C_ = torch.ones_like(A_) - 2 * A_ - torch.eye(A_.shape[0],device=device)
    S_ = torch.zeros_like(A_, requires_grad= True)
    delta = torch.zeros_like(x_2, device=device, requires_grad=True)
    adj_1 = edge2adj(x_1, edge_index_1)
    model.to(device)
    for epoch in range(steps): 
        A_prime = A_ + (S_ * C_)
        adj_hat = normalize_adj_tensor(A_prime + torch.eye(n_node,device=device))
        z1 = model(x_1, adj_1)
        z2 = model(x_2 + delta, adj_hat) 
        loss, _ = model.loss(z1, z2, batch_size=0) 
        attack_loss = loss.mean()
        attack_loss.backward()
        S_.data = (S_.data + alpha*S_.grad.detach().sign())
        S_.data = bisection(S_.data, eps, xi) # clip S
        S_.grad.zero_()
        
        delta.data = (delta.data + beta*delta.grad.detach().sign()).clamp(-0.5,0.5)        
        delta.grad.zero_()
        

    if not torch.equal(S_, S_.transpose(0,1)):
        lower_triangle_S_ = torch.tril(S_)
        eye_S_ = torch.diag(torch.diagonal(S_,0))
        pure_lower_triangle_S_ = lower_triangle_S_- eye_S_
        S_ = pure_lower_triangle_S_ + torch.transpose(pure_lower_triangle_S_,0,1)
     
    randm = torch.rand(n_node, n_node,device=device)
    discretized_S = torch.where(S_ > randm, torch.ones(n_node, n_node,device=device), torch.zeros(n_node, n_node, device=device))
    A_hat = A_ + discretized_S * C_ + torch.eye(n_node,device=device)
        
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    x_hat = x_2 + delta.data.to(device)
    return normalize_adj_tensor(A_hat), x_hat