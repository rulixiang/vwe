import argparse
import os
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='4,7', type=str, help="gpu")
parser.add_argument("--config", default='configs/voc.yaml', type=str, help="config")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import voc
from network import resnet_cam_mbk
from utils import pyutils, torchutils

import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return True

def makedirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
    return True

def validate(model=None, data_loader=None, codebook=None):

    print('Validating...')

    val_loss_meter = pyutils.AverageMeter('loss')
    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100,):
            inputs, labels = data['img'], data['label'].cuda()

            #outputs = model(inputs)
            outputs, x_hist, x_word, y_word, xx, _ = model(inputs, codebook=codebook)
            # forward + backward + optimize
            loss1 = F.multilabel_soft_margin_loss(outputs, labels)
            loss2 = F.multilabel_soft_margin_loss(x_hist, labels)
            loss3 = F.multilabel_soft_margin_loss(x_word, y_word)
            #loss4 = loss_re.mean()

            #val_loss_meter.add({'loss': loss1.item()})

    model.train()
    print('val loss1: %f, loss2: %f, loss3: %f\n'%(loss1, loss2, loss3))
    return True

def _get_entropy(logits: torch.Tensor) -> torch.Tensor:
    r"""Compute entropy according to the definition.

    Args:
        logits: Unscaled log probabilities.

    Return:
        A tensor containing the Shannon entropy in the last dimension.
    """
    probs = F.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, -1)
    return entropy

def train(config=None):
    # loop over the dataset multiple times

    num_workers = os.cpu_count()//2
    
    train_dataset = voc.VOC12ClassificationDataset(config.train.split, voc12_root=config.dataset.root_dir, resize_long=(320, 640), hor_flip=True, crop_size=512, crop_method="random")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    #max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc.VOC12ClassificationDataset(config.val.split, voc12_root=config.dataset.root_dir, crop_size=512)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
        for k in range(torch.cuda.device_count()):
            print('    %s: %s'%(args.gpu.split(',')[k], torch.cuda.get_device_name(k)))
    else:
        print('Using CPU:')
        device = torch.device('cpu')

    # build and initialize model
    model = resnet_cam_mbk.Net(n_classes=config.dataset.n_classes, backbone=config.exp.backbone, k_words=config.train.k_words)

    # save model to tensorboard 
    #writer_path = os.path.join(config.exp.backbone, config.exp.tensorboard_dir, TIMESTAMP)
    #writer = SummaryWriter(writer_path)
    #dummy_input = torch.rand(4, 3, 512, 512)
    #writer.add_graph(model, dummy_input)

    max_step = len(train_loader)*config.train.max_epochs
    param_groups = model.trainable_parameters()

    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': config.train.opt.learning_rate, 'weight_decay': config.train.opt.weight_decay},
        {'params': param_groups[1], 'lr': 10*config.train.opt.learning_rate, 'weight_decay': config.train.opt.weight_decay},
    ], lr=config.train.opt.learning_rate, weight_decay=config.train.opt.weight_decay, max_step=max_step)
    

    model = nn.DataParallel(model)
    model.train()
    model.to(device)
    
    makedirs(os.path.join(config.exp.backbone, config.exp.checkpoint_dir))
    makedirs(os.path.join(config.exp.backbone, config.exp.tensorboard_dir))
    
    iteration = 0
    train_loss_meter = pyutils.AverageMeter()

    codebook = torch.Tensor(config.train.k_words, 2048)
    nn.init.kaiming_normal_(codebook, a=np.sqrt(5))
    rho = 1e-3
    flag = True

    for epoch in range(config.train.max_epochs):

        print('Training epoch %d / %d ...'%(epoch+1, config.train.max_epochs))

        for _, data in tqdm(enumerate(train_loader), ncols=100, total=len(train_loader),):
        #for _, data in enumerate(train_loader):

            inputs, labels = data['img'], data['label'].cuda()
            inputs =  inputs.to(device)
            labels = labels.to(device)

            outputs, x_hist, x_word, y_word, xx, x_label = model(inputs, codebook=codebook.repeat(torch.cuda.device_count(), 1))
            #===============================#
            xx = xx.permute(0,2,3,1).contiguous().view(-1, xx.shape[1]).detach()
            yy = F.one_hot(x_label, num_classes=config.train.k_words)
            yy = yy.view(-1, yy.shape[-1]).type(torch.float)
            yy_sum = yy.sum(0) + 1e-4
            yy = yy / yy_sum

            ctr = torch.matmul(yy.T, xx)
            codebook = rho*ctr.cpu() + (1-rho)*codebook
            #==============================#
            if flag:
                codebook = xx[torch.randint(xx.size(0), (config.train.k_words,)),:]
                codebook = codebook.cpu()
                flag = False
            # forward + backward + optimize
            loss1 = F.multilabel_soft_margin_loss(outputs, labels)
            loss2 = F.multilabel_soft_margin_loss(x_hist, labels)
            loss3 = F.multilabel_soft_margin_loss(x_word, y_word)
            #loss4 = loss_re.mean()

            loss = loss1 + loss3
            
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            #running_loss += loss.item()
            train_loss_meter.add({'loss1':loss1.item(),'loss2':loss2.item(),'loss3':loss3.item()})

            iteration += 1

        #train_loss = train_loss_meter.pop('loss1')
        print('train loss1: %f, loss2: %f, loss3: %f,'%(train_loss_meter.pop('loss1'), train_loss_meter.pop('loss2'), train_loss_meter.pop('loss3')))
        #validate(model=model, data_loader=val_loader, codebook=codebook.repeat(torch.cuda.device_count(), 1))

        #writer.add_scalars("loss", {'train':train_loss, 'val':val_loss} , global_step=epoch)
        #writer.add_scalar("val/acc", scalar_value=score['Pixel Accuracy'], global_step=epoch)
        #writer.add_scalar("val/miou", scalar_value=score['Mean IoU'], global_step=epoch)
        dst_path = os.path.join(config.exp.backbone, config.exp.checkpoint_dir, config.exp.final_weights)
        dst_path = dst_path.replace('.pth', '_ep_'+str(epoch)+'.pth')
        torch.save(model.state_dict(), dst_path)
        np.save(dst_path.replace('.pth', '_codebook.npy'), codebook)
        print('model saved to %s...\n'%(dst_path))

    #dst_path = os.path.join(config.exp.backbone, config.exp.checkpoint_dir, config.exp.final_weights)
    #torch.save(model.state_dict(), dst_path)
    torch.cuda.empty_cache()

    return True

if __name__=="__main__":
    setup_seed(1)
    config = OmegaConf.load(args.config)
    print('configs: %s'%config)
    train(config)