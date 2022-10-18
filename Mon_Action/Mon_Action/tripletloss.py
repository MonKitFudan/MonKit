import torch
from torch import nn
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, labels):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        N = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(N, N)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # 选出所有正负样本对
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())  # 两两组合， 取label相同的a-p
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())  # 两两组合， 取label不同的a-n

        list_ap, list_an = [], []
        # 取出所有正样本对和负样本对的距离值
        for i in range(N):
            list_ap.append(dist[i][is_pos[i]].max().unsqueeze(0))
            list_an.append(dist[i][is_neg[i]].min().unsqueeze(0))
            dist_ap = torch.cat(list_ap)  # 将list里的tensor拼接成新的tensor
            dist_an = torch.cat(list_an)
        return dist_ap, dist_an

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss