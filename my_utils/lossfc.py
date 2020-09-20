import torch
from torch import nn


class LabelSmoothing(nn.Module):
    def __init__(self, device, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.weight = torch.FloatTensor([1.0, 1.0, 1.0, 1.0]).to(device)

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            # nll_loss = self.weight * nll_loss

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()

        else:
            return torch.nn.functional.cross_entropy(x, target)


class F1Loss(nn.Module):
    def __init__(self):
        super(F1Loss, self).__init__()
        pass

    def forward(self, x, target):
        x = x.softmax(dim=-1)
        calc1 = 2 * ((1 - x) * x) * target
        calc2 = ((1 - x) * x) + target
        loss = calc1 / calc2
        loss = loss.sum(dim=-1)
        loss = (1 - loss).mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.eps = 1e-8
        self.gamma = 1

    def forward(self, x, target):
        logit = x.softmax(dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * target * torch.log(logit)
        loss = loss * (1 - logit) ** self.gamma
        return loss.sum(dim=-1).mean()


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        pass

    def forward(self, x, target):
        outputs = x.log_softmax(dim=-1)

        loss = (- torch.sum(target * outputs, dim=1) + torch.sum(target * torch.log(target), dim=1)).mean()
        return loss

# if __name__ == "__main__":
#     x = torch.randn((2,2))
#     target = torch.LongTensor([0,0])
#     for lossfc in [LabelSmoothing("cpu"),F1Loss(),FocalLoss()]:
#         print(lossfc(x,target))
#
#     x = torch.randn((2,2))
#     target = torch.FloatTensor([[0.2,0.8],[0.3,0.7]])
#
#     for lossfc in [KLLoss()]:
#         print(lossfc(x,target))