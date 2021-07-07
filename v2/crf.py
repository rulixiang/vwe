
import argparse
import json
import os
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default='/home/rlx/VOCdevkit/VOC2012/', type=str, help="root_dir")
parser.add_argument("--txt_dir", default='dataset/voc/', type=str, help="txt_dir")
parser.add_argument("--split", default='train', type=str, help="split")
parser.add_argument("--cam_path", default='./resnet101/cam', type=str, help="logit")
parser.add_argument("--dst_dir", default='./cam_crf/', type=str, help="dst")
parser.add_argument("--bkgscore", default=0.15, type=float, help='bkgscore')


import torch
import torch.nn.functional as F
import numpy as np
from torch import multiprocessing
from scipy import misc
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class=21):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16),:]

def crf_proc(args):
    print("crf post-processing...")

    txt_name = os.path.join(args.txt_dir, args.split) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.root_dir, 'JPEGImages',)
    labels_path = os.path.join(args.root_dir, 'SegmentationClassAug')
    cam_path = args.cam_path
    crf_score_path = os.path.join(args.dst_dir, 'score')
    crf_pred_path = os.path.join(args.dst_dir, args.split, 'pred')
    crf_pred_rgb_path = os.path.join(args.dst_dir, args.split, 'pred_rgb')
    #mean_bgr = config.dataset.mean_bgr
    os.makedirs(crf_score_path, exist_ok=True)
    os.makedirs(crf_pred_path, exist_ok=True)
    os.makedirs(crf_pred_rgb_path, exist_ok=True)

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=1,   # 3
        pos_w=3,        # 3
        bi_xy_std=64,  # 121, 140 ok
        bi_rgb_std=3,   # 5, 5
        bi_w=4,         # 4, 5
    )

    def _job(i):

        name = name_list[i]

        #logit_name = os.path.join(logits_path, name + ".npy")
        #logit = np.load(logit_name)
        ##
        cam_name = os.path.join(cam_path, name + ".npy")
        cam_dict = np.load(cam_name, allow_pickle=True).item()
        logit = cam_dict['high_res']
        #keys = cam_dict['keys']+1
        lgt = np.zeros((1, logit.shape[0]+1, logit.shape[1], logit.shape[2]), dtype=np.float)
        lgt[0,0,:,:] = args.bkgscore
        lgt[0,1:,:,:] = logit

        logit = lgt
        ##print(logit.shape)
        ##

        image_name = os.path.join(images_path, name + ".jpg")
        image = misc.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        label = misc.imread(label_name)

        #image[:,:,0] = image[:,:,0] - mean_bgr[2]
        #image[:,:,1] = image[:,:,1] - mean_bgr[1]
        #image[:,:,2] = image[:,:,2] - mean_bgr[0]
        #image = image[:,:,[2,1,0]]

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        #####

        #####
        pred = np.argmax(prob, axis=0)

        #####
        keys = torch.zeros(size=[cam_dict['keys'].shape[0]+1])
        #np.insert(cam_dict['keys'], 0, )
        keys[1:]=cam_dict['keys']+1
        pred =keys[pred].numpy()
        ####

        _pred = np.squeeze(pred).astype(np.uint8)
        _pred_cmap = encode_cmap(_pred)

        misc.imsave(crf_pred_path+'/'+name+'.png', _pred)
        misc.imsave(crf_pred_rgb_path+'/'+name+'.png', _pred_cmap)

        return _pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    score = scores(gts, preds)
    json_path = os.path.join(crf_score_path, args.split) + '_crf.json'
    with open(json_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)
        
    print('Prediction results saved to %s.'%(crf_pred_path))
    print('Evaluation results saved to %s, pixel acc is %f, mean IoU is %f.'%(json_path, score['Pixel Accuracy'], score['Mean IoU']))
    
    return True

if __name__=="__main__":
    args = parser.parse_args()
    
    crf_score = crf_proc(args)
