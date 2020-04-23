from torch.nn import init
import numpy as np
import random
import torch

def to_tensor(x):
    return torch.from_numpy(x)

def to_numpy(x):
    return x.detach().cpu().numpy()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(m,norm_type="normal"):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if norm_type == "normal":
            init.normal_(m.weight, 0, 0.02)
        elif norm_type == "xavier":
            init.xavier_uniform_(m.weight)
        elif norm_type == "orthogonal":
            init.orthogonal_(m.weight,np.sqrt(2))

    elif classname.find('Linear') != -1:

        if norm_type == "normal":
            init.normal_(m.weight, 0, 0.02)
        elif norm_type == "xavier":
            init.xavier_uniform_(m.weight)
        elif norm_type == "orthogonal":
            init.orthogonal_(m.weight,np.sqrt(2))

