import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
import network.resnet as resnet
import numpy as np

def spatial_pyramid_hybrid_pool(x, levels=[1,2,4]):
    n,c,h,w = x.shape
    gamma = 2
    x_p = gamma * F.adaptive_avg_pool2d(x, (1,1))
    for i in levels:
        pool = F.max_pool2d(x, kernel_size=(h//i, w//i), padding=0)
        x_p = x_p + F.adaptive_avg_pool2d(pool, (1,1))

    return x_p/(gamma+len(levels))


def _cosine(x, centroid, p_norm=2):

    x_norm = F.normalize(x, dim=1, p=p_norm).detach()
    w_norm = F.normalize(centroid, dim=1, p=p_norm)

    x_corr = F.conv2d(x_norm, w_norm,)
    x_corr = F.softmax(x_corr, dim=1)
    x_label = x_corr.argmax(dim=1, keepdim=False)

    y_word = F.one_hot(x_label, num_classes=centroid.shape[0]).sum(dim=[1,2])>0 
    x_hist = torch.sum(x_corr, [2,3], keepdim=True)

    return x_corr, x_hist, y_word.detach()

class VWE(nn.Module):
    def __init__(self, k_words=256):
        super(VWE, self).__init__()
        
        self.k_words = k_words
        self.rho = 0.01

        self.centroid = nn.Parameter(torch.Tensor(self.k_words, 2048,), requires_grad=False,)
        #self.centroid = torch.Tensor(self.k_words, 2048)
        #nn.init.kaiming_normal_(self.centroid, a=np.sqrt(5))
        self.init = True
        #nn.init.xavier_uniform_(self.centroid)

    def forward(self, x):

        xx = x.permute(0,2,3,1).contiguous().view(-1, x.shape[1]).detach()

        if self.init:
            self.centroid = xx[torch.randint(xx.size(0), (self.k_words,)),:]
            #nn.init.kaiming_normal_(self.centroid, a=np.sqrt(5))
            self.init = False

        x_label, x_hist, y_word = _cosine(x=x.detach(), centroid=self.centroid.unsqueeze(-1).unsqueeze(-1))
        #======== update centroid ========#
        yy = F.one_hot(x_label, num_classes=self.k_words)
        yy = yy.view(-1, yy.shape[-1]).type(torch.float)
        yy_sum = yy.sum(0) + 1e-4
        yy = yy / yy_sum

        ctr = torch.matmul(yy.T, xx)
        self.centroid = self.rho * ctr + (1 - self.rho) * self.centroid
        #=================================#
        return x_label, x_hist, y_word 

class Net(nn.Module):

    def __init__(self, backbone='resnet50', n_classes=20, k_words=256):
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
        #self.codebook = torch.Tensor(size=(self.k_words, 2048))
        #self.VWE = VWE(k_words=self.k_words)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.fc_a, self.fc_b])

    def forward(self, x, return_cam=False, codebook=None, return_words=False):

        x1 = self.stage1(x)
        x2 = self.stage2(x1)#.detach()

        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        #x = torchutils.gap2d(x, keepdims=True)
        #out = F.adaptive_avg_pool2d(x4, (1,1))
        out = spatial_pyramid_hybrid_pool(x4)
        out = self.classifier(out)
        out = out.view(-1, self.n_classes)
        if return_cam:
            cam = F.conv2d(x4, self.classifier.weight)
            cam = F.relu(cam)
            return out, cam
        #x_label, x_hist, y_word = self.VWE(x=x4)
        ###
        codebook = codebook.unsqueeze(-1).unsqueeze(-1).to(x4.device)
        x_label, x_hist, y_word = _cosine(x=x4.detach(), centroid=codebook)

        if return_words:
            return codebook, x_label, x4[0]
        ###

        x_hist = self.fc_a(x_hist)
        x_hist = x_hist.view(-1, self.n_classes)

        x_word = F.adaptive_avg_pool2d(x4, (1,1))
        x_word = self.fc_b(x_word)
        x_word = x_word.view(-1, self.k_words)

        #====== reconstruction loss ======#
        #x4_reconstr = self.VWE.centroid[x_label,:]
        #x4_reconstr = x4_reconstr.permute(0,3,1,2)
        #loss_re = F.mse_loss(x4_reconstr, x4.detach())
        #=================================#

        return out, x_hist, x_word, y_word.type(torch.float32), x4, x_label

    def train(self, mode=True):
        for p in self.resnet.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
