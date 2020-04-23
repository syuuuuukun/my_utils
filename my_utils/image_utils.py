import torch
import torch.nn as nn
from torch.functional import F

import pandas as pd
import numpy as np
from PIL import Image
import cv2

def dataset_choice(n=50000,root_path = "",conf=0.95,small_data=False):
    url = "https://raw.githubusercontent.com/grapeot/Danbooru2018AnimeCharacterRecognitionDataset/master/faces.tsv"
    tagid = pd.read_csv(url, sep="\t", names=["filename", "tag", "x1", "y1", "x2", "y2", "prob"])
    new_tag_id = tagid[tagid["x1"].apply(lambda x: float(x.split()[-1])) > conf][["filename", "tag"]].reset_index(drop=True)
    img_path = list(new_tag_id.sample(n=n)["filename"])
    img_path = list(map(lambda x:"/".join(x.split("\\"))[1:],img_path))

    train_data_128,train_data_64,train_data_32,train_data_16,train_data_8,train_data_4 = [],[],[],[],[],[]
    for i, path in enumerate(img_path):
        if (i % 500) == 0:
            print(f"{i}")
        try:
            img = np.array(Image.open(root_path + path).convert("RGB"))
            train_data_128.append(img[None, :, :, :])
            if small_data:
                train_data_64.append(cv2.resize(img, (64,64), interpolation=cv2.INTER_LINEAR))
                train_data_32.append(cv2.resize(img, (32,32), interpolation=cv2.INTER_LINEAR))
                train_data_16.append(cv2.resize(img, (16,16), interpolation=cv2.INTER_LINEAR))
                train_data_8.append(cv2.resize(img, (8, 8) , interpolation=cv2.INTER_LINEAR))
                train_data_4.append(cv2.resize(img, (4, 4) , interpolation=cv2.INTER_LINEAR))
        except:
            pass
    return train_data_128,train_data_64,train_data_32,train_data_16,train_data_8,train_data_4

def gan_img_norm(x):
    return (x-127.5) / 127.5

def gan_img_renorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def gp_loss(fake,real,model_D,label=None,embed=None,epsilon=1e-3):
    b, c, h, w = fake.shape
    epsilon = torch.rand(b, 1, 1, 1, dtype=fake.dtype, device=fake.device)
    intpl = epsilon * fake + (1 - epsilon) * real
    intpl.requires_grad_()
    if label is None:
        f = model_D.forward(intpl)
        grad = torch.autograd.grad(f[1].sum(), intpl, create_graph=True)[0]
    else:
        f = model_D.forward(intpl, embed(label))
        grad = torch.autograd.grad(f.sum(), intpl, create_graph=True)[0]
    grad_norm = grad.view(b, -1).norm(dim=1)

    ##zero_centered_gp
    loss_gp = 10 * (grad_norm**2).mean()
    return loss_gp

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        #         self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, W,H = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, W*H).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, W*H)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, W*H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, W,H)

        out = self.gamma * out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.bn1 = nn.GroupNorm(out_channels // 8, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # self.bn2 = nn.GroupNorm(out_channels // 16, out_channels)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = torch.add(out, residual)
        out = self.relu2(out)
        return out

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1)
def deconv3x3(in_channels, out_channels, stride=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0,output_padding=1)