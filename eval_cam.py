import argparse
import os

from omegaconf import OmegaConf
from utils import pyutils
parser = argparse.ArgumentParser()
parser.add_argument("--crf", default=False, type=bool, help="crf post-processing")
parser.add_argument("--type", default='npy', type=str, help="file type")
parser.add_argument("--config", default='configs/voc.yaml', type=str, help="config")
parser.add_argument("--eval_set", default='train', type=str, help="eval set")
args = parser.parse_args()
from scipy import misc
import numpy as np
from tqdm import tqdm

def load_txt(txt_name):
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]
        return name_list

if __name__=="__main__":
    config = OmegaConf.load(args.config)
    print('\nEvaluating:')
    txt_name = config.cam.split
    eval_list = load_txt(txt_name)
    npy_dir = os.path.join(config.exp.backbone, config.exp.cam_dir)
    label_dir = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')
    
    preds = []
    labels = []

    for i in tqdm(eval_list, total=len(eval_list), ncols=100,):
        npy_name = os.path.join(npy_dir, i) + '.npy'
        cam_dict = np.load(npy_name, allow_pickle=True).item()
        label = misc.imread(os.path.join(label_dir, i) + '.png')


        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=config.cam.bkgscore)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]

        preds.append(cls_labels.copy())
        labels.append(label)

    scores = pyutils.scores(label_preds=preds, label_trues=labels)
    print('')
    print(scores)