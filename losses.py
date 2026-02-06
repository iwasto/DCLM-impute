import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConstrastiveLoss(nn.Module):
    def __init__(self, cells_num, temperature):
        super(ConstrastiveLoss, self).__init__()
        self.cells_num = cells_num
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(cells_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, cells_num):
        N = 2 * cells_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(cells_num):
            mask[i, cells_num + i] = 0
            mask[cells_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.cells_num
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.cells_num)
        sim_j_i = torch.diag(sim, -self.cells_num)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

def _nan2zero(x):    #将输入张量x中的NaN值替换为0
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)   

def _nan2inf(x):    #将输入张量x中的NaN值替换为无穷大
    return torch.where(torch.isnan(x), torch.zeros_like(x)+np.inf, x)  

def _nelem(x):      #计算输入张量 x 中非 NaN 元素的数量，并确保结果不为 0。
    nelem = torch.sum(torch.tensor(~torch.isnan(x),dtype = torch.float32))  
    return torch.tensor(torch.where(torch.equal(nelem, 0.), 1., nelem), dtype = x.dtype)


def _reduce_mean(x):    #计算输入张量 x 的平均值，同时处理 NaN 值。
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.divide(torch.sum(x), nelem)


def mse_loss(y_true, y_pred):  #计算均方误差（MSE）损失。
    ret = torch.square(y_pred - y_true)

    return _reduce_mean(ret)

 
def poisson_loss(y_true, y_pred):  #计算泊松分布的损失函数
    y_pred = torch.tensor(y_pred, dtype = torch.float32)
    y_true = torch.tensor(y_true, dtype = torch.float32)
    nelem = _nelem(y_true)
    y_true = _nan2zero(y_true)
    ret = y_pred - y_true*torch.log(y_pred+1e-10) + torch.lgamma(y_true+1.0)

    return torch.divide(torch.sum(ret), nelem)



class NB(object):  #计算负二项分布损失
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        y_true = torch.tensor(y_true, dtype = torch.float32)
        y_pred = torch.tensor(y_pred, dtype = torch.float32) * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

            # Clip theta
        theta = torch.minimum(self.theta,torch.tensor(1e6))

        t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))


        final = t1 + t2

        final = _nan2inf(final)

        if mean:
            if self.masking:
                final = torch.divide(torch.sum(final), nelem)
            else:
                final = torch.sum(final)


        return final

class ZINB(NB):    #ZINB继承自NB类，并扩展了负二项分布（Negative Binomial，NB）损失函数，以实现零膨胀负二项分布（Zero-Inflated Negative Binomial，ZINB）损失函数
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps


            # reuse existing NB neg.log.lik.
            # mean is always False here, because everything is calculated
            # element-wise. we take the mean only in the end
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0-self.pi+eps)

        y_true = torch.tensor(y_true, dtype = torch.float32)
        y_pred = torch.tensor(y_pred, dtype = torch.float32) * scale_factor
        theta = torch.minimum(self.theta,torch.tensor(1e6))

        zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
        result = torch.where(torch.less(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda*torch.square(self.pi)
        result += ridge

        if mean:
            if self.masking:
                result = _reduce_mean(result)
            else:
                result = torch.sum(result)

        result = _nan2inf(result)

        return result


def _cosine_loss(pred, target):
    """计算余弦相似度损失"""
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return 2 - 2 * (pred_norm * target_norm).sum(dim=-1).mean()

