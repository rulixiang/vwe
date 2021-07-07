import torch
import torch.nn as nn
import torch.nn.functional as F
from network import resnet
import numpy as np

def spatial_pyramid_hybrid_pool(x, levels=[1,2,4]):
    n,c,h,w = x.shape
    gamma = 0
    x_p = gamma * F.adaptive_avg_pool2d(x, (1,1))
    for i in levels:
        pool = F.max_pool2d(x, kernel_size=(h//i, w//i), padding=0)
        x_p = x_p + F.adaptive_avg_pool2d(pool, (1,1))

    return x_p/(gamma+len(levels))

def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def decov_loss(inputs=None):
    '''
    inputs: K*D*1*1
    '''
    #inputs = F.normalize(inputs, dim=1, p=2)
    inputs_s = inputs.squeeze()
    #inputs_cov = inputs_s.mm(inputs_s.T)

    inputs_mean = inputs_s - torch.mean(inputs_s, dim=1, keepdim=True)
    inputs_cov = inputs_mean.mm(inputs_mean.T)# / float(inputs_s.shape[0])
    #======
    loss = torch.norm(inputs_cov, p='fro') - (torch.diag(inputs_cov)**2).sum().sqrt()

    return 0.5*loss

def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    r"""Compute entropy according to the definition.

    Args:
        logits: Unscaled log probabilities.

    Return:
        A tensor containing the Shannon entropy in the last dimension.
    """
    probs = F.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, -1)
    return entropy.mean()


def _cosine(x, centroid, p_norm=2, tau=1.0):
    x_norm = F.normalize(x, dim=1, p=p_norm).detach()
    w_norm = F.normalize(centroid, dim=1, p=p_norm)

    x_corr = F.conv2d(x_norm, w_norm,)
    x_corr = F.softmax(x_corr*tau, dim=1)

    y_word = F.one_hot(torch.argmax(x_corr, dim=1), num_classes=centroid.shape[0]).sum(dim=[1,2])>0 
    x_hist = torch.sum(x_corr, [2,3], keepdim=True)

    return x_corr, x_hist, y_word.detach()

class VWE(nn.Module):
    def __init__(self, k_words=None):
        super(VWE, self).__init__()
        
        self.k_words = k_words

        self.centroid = nn.Parameter(torch.Tensor(self.k_words, 2048, 1, 1), requires_grad=True)
        nn.init.kaiming_normal_(self.centroid, a=np.sqrt(5))
        #nn.init.xavier_uniform_(self.centroid)

    def forward(self, x):
        x_corr, x_hist, y_word = _cosine(x=x, centroid=self.centroid)
        return x_corr, x_hist, y_word 

class Net(nn.Module):

    def __init__(self, backbone='resnet101', n_classes=20, k_words=None):
        super(Net, self).__init__()

        if backbone=='resnet50':
            self.resnet = resnet.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        elif backbone=='resnet101':
            self.resnet = resnet.resnet101(pretrained=True, strides=(2, 2, 2, 1))

        self.n_classes = n_classes
        self.k_words = k_words

        self.stage1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool, self.resnet.layer1)
        self.stage2 = nn.Sequential(self.resnet.layer2)
        self.stage3 = nn.Sequential(self.resnet.layer3)
        self.stage4 = nn.Sequential(self.resnet.layer4)

        self.classifier = nn.Conv2d(2048, self.n_classes, 1, bias=False)
        self.fc_a = nn.Conv2d(self.k_words, self.n_classes, 1, bias=False)
        self.fc_b = nn.Conv2d(2048, self.k_words, 1, bias=False)
        self.VWE = VWE(k_words=self.k_words)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.VWE, self.fc_a, self.fc_b])

    def forward(self, x, return_cam=False):

        x1 = self.stage1(x)
        x2 = self.stage2(x1).detach()

        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        #x = torchutils.gap2d(x, keepdims=True)
        #out = F.adaptive_avg_pool2d(x4, (1,1))
        out = spatial_pyramid_hybrid_pool(x4)
        out = self.classifier(out)
        out = out.view(-1, self.n_classes)

        x_corr, x_hist, y_word = self.VWE(x=x4)
        x_hist = self.fc_a(x_hist)
        x_hist = x_hist.view(-1, self.n_classes)

        x_word = F.adaptive_avg_pool2d(x4, (1,1))
        x_word = self.fc_b(x_word)
        x_word = x_word.view(-1, self.k_words)

        if return_cam:
            cam = F.conv2d(x4, self.classifier.weight)
            cam = F.relu(cam)
            return out, cam

        #======= decov + entropy =======#
        loss_decov = decov_loss(self.VWE.centroid)
        loss_entropy = entropy_loss(x_word)

        #===============================#

        return out, x_hist, x_word, y_word.type(torch.float32), loss_decov, loss_entropy

    def train(self, mode=True):
        for p in self.resnet.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
