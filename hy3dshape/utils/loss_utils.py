import torch
import torch.nn.functional as F
import torch.nn as nn
def chamfer_distance_with_average(p1, p2):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)
    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)
    p2 = p2.repeat(p1.size(0), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist_norm = torch.norm(dist, 2, dim=2)
    dist1 = torch.min(dist_norm, dim=1)[0]
    dist2 = torch.min(dist_norm, dim=0)[0]
    loss = 0.5 * ((torch.mean(dist1)) + (torch.mean(dist2)))
    return loss

def cross_entropy_with_probs_batch(input, target, weight=None, reduction="mean"):  # tested, same as nn.CrossEntropyLoss at dim=1, CE can be negative
    # input_logsoftmax = F.log_softmax(input, dim=2)
    input_logsoftmax = torch.log(input+1e-6)
    cum_losses = -target * input_logsoftmax
    if weight is not None:
        cum_losses = cum_losses * weight.unsqueeze(1)  # Broadcasting the weight

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.sum(dim=2).mean(dim=1).mean(dim=0)
    elif reduction == "sum":
        return cum_losses.sum(dim=2).sum(dim=1).mean(dim=0)
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")
    
def cos_loss(input, target):
    # input = F.softmax(input, dim=-1)
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    similarity = cos(input, target)
    loss = 1 - similarity.mean()  
    return loss

def cos_loss_clamp(input, target):
    # input = F.softmax(input, dim=-1)*(1 + 2*0.001) - 0.001
    # import pdb;pdb.set_trace()
    input = input*(1 + 2*0.001) - 0.001
    input = torch.clamp(input, 0, 1)
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    similarity = cos(input, target)
    loss = 1 - similarity.mean()  
    return loss