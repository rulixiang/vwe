import argparse
import os
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0,1,2', type=str, help="gpu")
parser.add_argument("--config", default='configs/voc.yaml', type=str, help="config")
parser.add_argument("--weights", default='resnet_cam_final_ep_6.pth', type=str, help="weight path")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
from tqdm import tqdm
import numpy as np
from dataset import voc
from network import resnet_cam_mbk
from collections import OrderedDict
from utils import imutils

def makedirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
    return True

def _infer_cam(pid, model=None, dataset=None, config=None):

    data_loader = torch.utils.data.DataLoader(dataset[pid], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    model.eval()
    cam_dir = os.path.join(config.exp.backbone, config.exp.cam_dir)
    makedirs(cam_dir)
    with torch.no_grad(), torch.cuda.device(pid):
        model.cuda()
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100,):
            img_name, input_list, labels, size_ = data['name'], data['img'], data['label'], data['size']

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)
            img_size = input_list[0].shape[-2:]
            strided_size = imutils.get_strided_size(size_, 4)
            cam_list = []
            for inputs in input_list:
                inputs = inputs[0].cuda(non_blocking=True)
                _, cams = model(inputs, return_cam=True)
                cams_ = torch.max(cams[0], cams[1].flip(-1))
                cam_list.append(cams_)
                #labels = labels.to(outputs.device)
            
            strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(cam, 0), strided_size, mode='bilinear', align_corners=False)[0] for cam in cam_list]), 0)

            resized_cam_list = [F.interpolate(cam.unsqueeze(0), img_size, mode='bilinear', align_corners=False)[0] for cam in cam_list]
            
            out_cam = torch.sum(torch.stack(resized_cam_list, dim=0), dim=0)

            valid_label = torch.nonzero(labels[0])[:,0]

            strided_cam = strided_cam[valid_label]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            high_res_cam = out_cam[valid_label,:,:]
            high_res_cam /= F.adaptive_max_pool2d(high_res_cam, (1, 1)) + 1e-5
            #loss = F.multilabel_soft_margin_loss(outputs, labels)
            np.save(os.path.join(cam_dir, img_name[0] + '.npy'), {"keys": valid_label.cpu().numpy(), "cam": strided_cam.cpu().numpy(), "high_res": high_res_cam.cpu().numpy()})
    return None

def main(config=None):

    infer_dataset = voc.VOC12ClassificationDatasetMSF(voc12_root=config.dataset.root_dir, img_name_list_path=config.cam.split, scales=config.cam.scales)

    n_gpus = torch.cuda.device_count()

    split_dataset = [torch.utils.data.Subset(infer_dataset, np.arange(i, len(infer_dataset), n_gpus)) for i in range (n_gpus)]

    # device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build and initialize model
    model = resnet_cam_mbk.Net(n_classes=config.dataset.n_classes, backbone=config.exp.backbone)
    model_path = os.path.join(config.exp.backbone, config.exp.checkpoint_dir, args.weights)

    #model = nn.DataParallel(model)
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()

    #_infer_cam(model=)
    print('Inferring...')

    makedirs(os.path.join(config.exp.backbone, config.exp.cam_dir))

    multiprocessing.spawn(_infer_cam, nprocs=n_gpus, args=(model, split_dataset, config), join=True)

    torch.cuda.empty_cache()

    return True

if __name__=="__main__":

    config = OmegaConf.load(args.config)
    main(config)