#CUDA_VISIBLE_DEVICES=5 python main.py --mode test --backbone resnet --dataset drive --resume run/drive/unet/experiment_2/checkpoint_100.pth.tar --gpu-ids 0 --loss-type bce
CUDA_VISIBLE_DEVICES=1 python main.py --mode test --model_name unet --backbone resnet --dataset drive --resume run/drive/focal/experiment_8/checkpoint_120.pth.tar --gpu-ids 0 --loss-type bce --sh 5 --sw 5 --batch-size 128
