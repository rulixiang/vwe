gpus=4,5,6,7
logger=log.txt

python train_cam.py --gpu $gpus
python infer_cam.py --gpu $gpus
python eval_cam.py