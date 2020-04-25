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


class log_ploter(object):
    def __init__(self):
        self.names = None
        self.losses = []

    def ploter(self):
        print()
        for name, loss in zip(self.names, self.losses):
            print(f"{name}: {np.mean(loss):.4}")
        self.losses = [[] for i in range(len(self.names))]

    def get_var_names(self, vars):
        names = []
        for var in vars:
            for k, v in globals().items():
                if id(v) == id(var):
                    names.append(k)
        return names

    def updater(self, losses):
        if self.names is None:
            self.names = self.get_var_names(losses)
            self.losses = [[] for i in range(len(self.names))]

        for i, loss in enumerate(losses):
            self.losses[i].append(loss.item())
