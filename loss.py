import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

class TcaLoss(nn.Module):
    def __init__(self, margin=0.7):
        super(TcaLoss, self).__init__()
        self.margin = margin
        self.rank_loss = nn.MarginRankingLoss(margin=0.2)

    def forward(self, inputs, targets):
        ft1, ft2, ft3, ft4 = torch.chunk(inputs, 4, 0)
        lb1, lb2, lb3, lb4 = torch.chunk(targets, 4, 0)

        feat1 = (ft1 + ft2) / 2
        feat2 = (ft1 + ft3) / 2
        feat3 = (ft1 + ft4) / 2
        feat4 = (ft2 + ft3) / 2
        feat5 = (ft2 + ft4) / 2
        feat6 = (ft3 + ft4) / 2

        lbs = lb1.unique()
        n = lbs.size(0)
        num = inputs.size(0)

        ft1 = feat1.chunk(n, 0)
        ft2 = feat2.chunk(n, 0)
        ft3 = feat3.chunk(n, 0)
        ft4 = feat4.chunk(n, 0)
        ft5 = feat5.chunk(n, 0)
        ft6 = feat6.chunk(n, 0)

        center1 = []
        center2 = []
        center3 = []
        center4 = []
        center5 = []
        center6 = []
        centers = []

        for i in range(n):
            center1.append(torch.mean(ft1[i], dim=0, keepdim=True))
            center2.append(torch.mean(ft2[i], dim=0, keepdim=True))
            center3.append(torch.mean(ft3[i], dim=0, keepdim=True))
            center4.append(torch.mean(ft4[i], dim=0, keepdim=True))
            center5.append(torch.mean(ft5[i], dim=0, keepdim=True))
            center6.append(torch.mean(ft6[i], dim=0, keepdim=True))

        for i in range(num):
            centers.append(inputs[targets == targets[i]].mean(0))

        ft1 = torch.cat(center1)
        ft2 = torch.cat(center2)
        ft3 = torch.cat(center3)
        ft4 = torch.cat(center4)
        ft5 = torch.cat(center5)
        ft6 = torch.cat(center6)
        centers = torch.stack(centers)

        dist_25 = pdist_torch(ft2, ft5) #D(vis-ir,Mvis-Mir)
        dist_16 = pdist_torch(ft1, ft6) #D(vis-Mvis,ir-Mir)
        dist_34 = pdist_torch(ft3, ft4) #D(vis-Mir,ir-Mvis)
        dist = pdist_torch(inputs,centers)

        # For each anchor, find the hardest positive and negative
        mask = lbs.expand(n, n).eq(lbs.expand(n, n).t())
        mask_1 = targets.expand(num, num).eq(targets.expand(num, num).t())
        dist_an, dist_ap = [], []
        dist_ap_25, dist_ap_16, dist_ap_34 = [], [], []

        for i in range(n):
            dist_ap_25.append(dist_25[i][mask[i]].min().unsqueeze(0))
            dist_ap_16.append(dist_16[i][mask[i]].min().unsqueeze(0))
            dist_ap_34.append(dist_34[i][mask[i]].min().unsqueeze(0))

        for i in range(num):
            dist_an.append((self.margin - dist[i][mask_1[i] == 0]).clamp(min=0.0).mean())

        dist_an = torch.stack(dist_an)
        dist_ap_25 = torch.cat(dist_ap_25)
        dist_ap_16 = torch.cat(dist_ap_16)
        dist_ap_34 = torch.cat(dist_ap_34)

        dist_ap_center = torch.stack((dist_ap_25, dist_ap_16, dist_ap_34), 0)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)

        loss_center, _ = torch.max(dist_ap_center, dim=0)
        loss_center = (loss_center.sum())
        loss = loss_center / 4 + dist_an.mean()
        return loss

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx