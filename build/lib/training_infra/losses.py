import torch
import torch.nn.functional as F
import pyro

def neg_sqrt_loss(model_preds, y):
    if 'p' in model_preds:
        p = model_preds['p']
    elif 'log_p' in model_preds:
        p = torch.exp(model_preds['log_p'])
    return -torch.sqrt(p[torch.arange(y.shape[0]),y]).mean()

def nll_loss(model_preds, y):
    if 'log_p' in model_preds:
        log_p = model_preds['log_p']
    elif 'p' in model_preds:
        log_p = torch.log(model_preds['p'])
    return F.nll_loss(log_p, y)

def multi_target_crossentropy_loss(model_preds,y):
    counts = y.sum(axis=-1, keepdim=True)
    target_p = y/counts
    if 'log_p' in model_preds:
        log_p = model_preds['log_p']
    elif 'p' in model_preds:
        log_p = torch.log(model_preds['p'])
    unreduced_loss = F.cross_entropy(log_p, target_p, reduction='none')
    return torch.dot(unreduced_loss, counts.view(-1))/counts.sum()

def multinomial_loss(model_preds,y):
    if 'log_p' in model_preds:
        log_p = model_preds['log_p']
    elif 'p' in model_preds:
        log_p = torch.log(model_preds['p'])
    counts = y.sum(axis=-1, keepdim=True)
    torch.lgamma(y+1).sum(axis=-1,keepdim=True)
    unreduced_loss = torch.lgamma(y+1).sum(axis=-1,keepdim=True) - torch.lgamma(counts+1) - (y*log_p).sum(axis=-1,keepdim=True)
    return unreduced_loss.sum()/counts.sum()

def dirichlet_multinomial_loss(model_preds,y):
    counts = y.sum(axis=-1)
    if 'alpha' in model_preds:
        alpha = model_preds['alpha']
    elif 'n' in model_preds:
        n = model_preds['n']
        if 'p' in model_preds:
            p = model_preds['p']
        elif 'log_p' in model_preds:
            p = torch.exp(model_preds['log_p'])
        alpha = n*p
    return -pyro.distributions.DirichletMultinomial(alpha, total_count=counts).log_prob(y).sum()/counts.sum()

loss_dict = {'neg_sqrt_loss': neg_sqrt_loss,
             'nll_loss': nll_loss,
             'multi_target_crossentropy_loss': multi_target_crossentropy_loss,
             'dirichlet_multinomial_loss': dirichlet_multinomial_loss,
             'multinomial_loss': multinomial_loss
             }