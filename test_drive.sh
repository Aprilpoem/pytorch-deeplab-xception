CUDA_VISIBLE_DEVICES=5 python main.py --mode test --backbone resnet --dataset drive --resume run/drive/unet/experiment_2/checkpoint_100.pth.tar --gpu-ids 0 --loss-type bce
