# Weakly-Supervised Semantic Segmentation with Visual Words Learning and Hybrid Pooling

Implementation of [Weakly-Supervised Semantic Segmentation with Visual Words Learning and Hybrid Pooling](https://lixiangru.cn/assets/files/vwl.pdf) [Under Review], an improved version of our [IJCAI 2021 work](https://lixiangru.cn/assets/files/vwe.pdf).

## What's New?

Compared with the conference version, this work further

- improves the learning-based strategy;
- proposes the memory-bank strategy;
- includes further experiments on more datasets;
- remarkably surpasses the latest state-of-the-art methods.

<img align="center" src="./figures/vwl.png"/>

## Start

### Create and activate conda environment

```bash
conda create --name py36 python=3.6
conda activate py36
pip install -r requirments.txt
```

### Clone this repo

```bash
git clone https://github.com/rulixiang/vwe.git
cd vwe/v2
```

### Train & Infer & Evaluate CAMs

- For the Learning-based startegy:

```bash
# train network
python train_cam.py --gpu 0 --configs/voc.yaml
# infer cam
python infer_cam.py --gpu 0 --configs/voc.yaml
# evaluate cam
python eval_cam.py
```

- For the memory-bank strategy:

```bash
# train network
python train_cam_mbk.py --gpu 0 --configs/voc.yaml
# infer cam
python infer_cam_mbk.py --gpu 0 --configs/voc.yaml
# evaluate cam
python eval_cam.py
```

## Results

- Our trained weights are available at [Google Drive](https://drive.google.com/drive/folders/1h8Erevo7uQLq56yP-c89za28a0NvxzWG?usp=sharing).
- CAMs on PASCAL VOC 2012 dataset:

| Method | train mIoU | val mIoU |                                              Weights                                               |
|:------:|:----------:|:--------:|:--------------------------------------------------------------------------------------------------:|
| VWL-M  |    56.9    |   56.4   | [Google Drive](https://drive.google.com/file/d/1S-jJiR35U_9a2IP2m9wdgLeotnQOje3Y/view?usp=sharing) |
| VWL-L  |    57.3    |   56.9   | [Google Drive](https://drive.google.com/file/d/1qxTJcodpxTGCK8NLKG3zY7F_bwVKS62E/view?usp=sharing) |

- To refine the initial CAMs, we use [IRNet](https://github.com/jiwoon-ahn/irn).

- On PASCAL VOC 2012 dataset:

| Method |  Seg Net  | Pretrain | val mIoU |                           test mIoU                            |                                              Weights                                               |
|:------:|:---------:|:--------:|:--------:|:--------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|
| VWL-M  | DeepLabV2 | ImageNet |   68.7   | [69.2](http://host.robots.ox.ac.uk:8080/anonymous/XJDOJG.html) | [Google Drive](https://drive.google.com/file/d/1UtkjpDk5hdS0lVWGXqdSffjmeQrn-E2V/view?usp=sharing) |
| VWL-M  | DeepLabV2 |   COCO   |   70.6   | [70.4](http://host.robots.ox.ac.uk:8080/anonymous/J00QBG.html) | [Google Drive](https://drive.google.com/file/d/1VGw4rKg3Ex4N-GzjGcGEZQJlev6cKXnS/view?usp=sharing) |
| VWL-L  | DeepLabV2 | ImageNet |   69.2   | [69.2](http://host.robots.ox.ac.uk:8080/anonymous/Y0XECB.html) | [Google Drive](https://drive.google.com/file/d/1tBY3nyuiO9DU6jR40ZABh4EIGq53SrOC/view?usp=sharing) |
| VWL-L  | DeepLabV2 |   COCO   |   70.6   | [70.7](http://host.robots.ox.ac.uk:8080/anonymous/0QVYDO.html) | [Google Drive](https://drive.google.com/file/d/1OrbpPmG5Q1OJr2qczBw13O1-hixGtdGh/view?usp=sharing) |
| VWL-L  |  EMANet   | ImageNet |   70.8   | [71.1](http://host.robots.ox.ac.uk:8080/anonymous/FJJDSP.html) |                                                 --                                                 |

- On MS COCO 2014 dataset:

| Method |  Seg Net  | val mIoU |
|:------:|:---------:|:--------:|
| VWL-M  | DeepLabV2 |   36.1   |
| VWL-L  | DeepLabV2 |   36.2   |

## Citation

```
@inproceedings{
  ru2021learning,
  title={Learning Visual Words for Weakly-Supervised Semantic Segmentation},
  author={Lixiang Ru and Bo Du and Chen Wu},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2021},
}

@article{
  ru2021weakly,
  title={Weakly-Supervised Semantic Segmentation with Visual Words Learning and Hybrid Pooling},
  author={Lixiang Ru and Bo Du and Yibing Zhan and Chen Wu},
  year={2021},
}
```
