import argparse
import os
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0,1,2', type=str, help="gpu")
parser.add_argument("--config", default='configs/voc.yaml', type=str, help="config")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import voc
from network import resnet_cam
from utils import pyutils, torchutils

def makedirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
    return True

def validate(model=None, data_loader=None,):

    print('Validating...')

    val_loss_meter = pyutils.AverageMeter('loss')
    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100,):
            inputs, labels = data['img'], data['label'].cuda()

            #outputs = model(inputs)
            outputs, x_hist, x_word, y_word = model(inputs)
            # forward + backward + optimize
            loss1 = F.multilabel_soft_margin_loss(outputs, labels)
            loss2 = F.multilabel_soft_margin_loss(x_hist, labels)
            loss3 = F.multilabel_soft_margin_loss(x_word, y_word)

            val_loss_meter.add({'loss': loss1.item()})

    model.train()

    return val_loss_meter.pop('loss')

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
    model = resnet_cam.Net(n_classes=config.dataset.n_classes, backbone=config.exp.backbone)

    # save model to tensorboard 
    writer_path = os.path.join(config.exp.backbone, config.exp.tensorboard_dir, TIMESTAMP)
    writer = SummaryWriter(writer_path)
    dummy_input = torch.rand(4, 3, 512, 512)
    writer.add_graph(model, dummy_input)

    max_step = len(train_loader)*config.train.max_epochs
    param_groups = model.trainable_parameters()
    '''
    optimizer = torch.optim.SGD(
        # 
        params=[
            {
                "params": param_groups[0],
                "lr": config.train.opt.learning_rate,
                "weight_decay": config.train.opt.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 10 * config.train.opt.learning_rate,
                "weight_decay": config.train.opt.weight_decay,
            },
        ],
        momentum=config.train.opt.momentum,
    )
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
    '''
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
    train_loss_meter = pyutils.AverageMeter('loss1','loss2','loss3')

    for epoch in range(config.train.max_epochs):

        print('Training epoch %d / %d ...'%(epoch+1, config.train.max_epochs))

        for _, data in tqdm(enumerate(train_loader), ncols=100, total=len(train_loader),):
        #for _, data in enumerate(train_loader):

            inputs, labels = data['img'], data['label'].cuda()
            inputs =  inputs.to(device)
            labels = labels.to(device)

            outputs, x_hist, x_word, y_word = model(inputs)
            
            # forward + backward + optimize
            loss1 = F.multilabel_soft_margin_loss(outputs, labels)
            loss2 = F.multilabel_soft_margin_loss(x_hist, labels)
            loss3 = F.multilabel_soft_margin_loss(x_word, y_word)

            loss = loss1 + loss2 + loss3
            
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            #running_loss += loss.item()
            train_loss_meter.add({'loss1':loss1.item(),'loss2':loss2.item(),'loss3':loss3.item(),})

            iteration += 1
            ## poly scheduler
            '''
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*(1 - float(iteration) / max_step) ** config.train.opt.power
            '''
        # save to tensorboard
        '''
        temp_k = 4
        inputs_part = inputs[0:temp_k,:]
        resized_outputs = F.interpolate(outputs, size=inputs.shape[2:], mode='bilinear', align_corners=True)
        outputs_part = resized_outputs[0:temp_k,:]
        labels_part = labels[0:temp_k,:]
        grid_inputs, grid_outputs, grid_labels = imutils.tensorboard_image(inputs=inputs_part, outputs=outputs_part, labels=labels_part, bgr=config.dataset.mean_bgr)
        writer.add_image("train/images", grid_inputs, global_step=epoch)
        writer.add_image("train/preds", grid_outputs, global_step=epoch)
        writer.add_image("train/labels", grid_labels, global_step=epoch)
        '''
        train_loss = train_loss_meter.pop('loss1')
        val_loss = validate(model=model, data_loader=val_loader)
        print('train loss: %f, val loss: %f\n'%(train_loss, val_loss))

        #writer.add_scalars("loss", {'train':train_loss, 'val':val_loss}, global_step=epoch)
        #writer.add_scalar("val/acc", scalar_value=score['Pixel Accuracy'], global_step=epoch)
        #writer.add_scalar("val/miou", scalar_value=score['Mean IoU'], global_step=epoch)

    dst_path = os.path.join(config.exp.backbone, config.exp.checkpoint_dir, config.exp.final_weights)
    torch.save(model.state_dict(), dst_path)
    torch.cuda.empty_cache()

    return True

if __name__=="__main__":

    config = OmegaConf.load(args.config)
    print('configs: %s'%config)
    train(config)