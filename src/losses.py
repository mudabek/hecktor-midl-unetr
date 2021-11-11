import torch
from torch import nn
from metrics import dice_scst, hausdorff


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


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.cross_entropy = torch.nn.BCELoss()#nn.CrossEntropyLoss()

    def forward(self, input, target):
        # import pdb
        # pdb.set_trace()
        # n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # if n_pred_ch == n_target_ch:
        #     # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
        #     target = torch.argmax(target, dim=1)
        # else:
        #     target = torch.squeeze(target, dim=1)
        # target = target.long()

        return self.cross_entropy(input, target)


class SCST(nn.Module):
    def __init__(self, metric='dice'):
        super(SCST, self).__init__()
        self.metric = metric 
        self.dice_loss = DiceLoss()
        # self.ce_loss = CELoss()
        self.focal_loss = FocalLoss()


    def forward(self, input, target, baseline_score, phase):
        if phase == 'train':
            # Calculate score and advantage for a given metric
            if self.metric == 'dice':
                score = dice_scst(input, target)
                advantage = score - baseline_score
            else:
                score = hausdorff(input, target)
                advantage = baseline_score - score  

            # Calculate log of probabilities
            # odds = torch.exp(input)
            # prob = torch.exp(input) / (1 + torch.exp(input))

            # Calculate loss
            # loss = -advantage[:, None, None, None, None] * torch.log(torch.exp(input) / (1 + torch.exp(input)))
            # loss = -advantage[:, None, None, None, None].data * torch.log(input)
            # ce_loss = self.ce_loss(input, target)
            focal_loss = self.focal_loss(input, target)
            loss = -advantage.mean() * focal_loss
            
            return loss.mean()
        return self.dice_loss(input, target)

class Dice_and_CELoss(nn.Module):
    def __init__(self):
        super(Dice_and_CELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss =  CELoss()

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.ce_loss(input, target)
        return loss

