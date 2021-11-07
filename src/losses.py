import torch
from torch import nn
from metrics import dice, hausdorff


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss



class SCST(nn.Module):
    def __init__(self, metric='dice'):
        super(SCST, self).__init__()
        self.metric = metric 


    def forward(self, input, target, baseline_score):

        # Calculate score and advantage for a given metric
        if self.metric == 'dice':
            score = dice(input, target)
            advantage = score - baseline_score
        else:
            score = hausdorff(input, target)
            advantage = baseline_score - score  

        # Calculate log of probabilities
        odds = torch.exp(input)
        prob = odds / (1 + odds)
        log_prob = torch.log(prob)

        # Calculate loss
        loss = -advantage * log_prob
        
        return loss.mean()
