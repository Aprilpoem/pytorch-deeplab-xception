#CUDA_VISIBLE_DEVICES=5 python main.py --mode test --backbone resnet --dataset drive --resume run/drive/unet/experiment_2/checkpoint_100.pth.tar --gpu-ids 0 --loss-type bce
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --model_name unet --backbone resnet --dataset drive --resume run/drive/unet0/experiment_6/checkpoint_200.pth.tar --gpu-ids 0 --loss-type bce --sh 5 --sw 5 --batch-size 128
