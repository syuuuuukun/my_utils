import numpy as np
import torch

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

##Cutmix
def cutmix(data,label,alpha):
    b = data.shape[0]
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(b)
    target_a = label
    target_b = label[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.shape, lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    return data,target_a,target_b,lam

def mixup(data,label,alpha):
    b = data.shape[0]
    lam = np.random.beta(alpha, alpha)
    lam = torch.from_numpy(lam).unsqueeze(-1).to(data.device)
    rand_index = torch.randperm(b)
    target_a = label
    target_b = label[rand_index]
    data_a = data
    data_b = data[rand_index]
    mix_data = (lam*data_a)+((1-lam)*data_b)
    return mix_data,target_a,target_b,lam

# if __name__ == "__main__":
#     data,label_a,label_b,lam = cutmix(data,label)
#     out = model(data)
#     loss = (lossfc(out,label_a)*lam) + (lossfc(out,label_b)*(1.-lam))