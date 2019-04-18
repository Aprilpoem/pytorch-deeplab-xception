#CUDA_VISIBLE_DEVICES=2 python main.py --model_name deeplab --backbone resnet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname deeplab-resnet --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type ce --pw 48 --ph 48 --npatches 120000 --inchannels 3
#CUDA_VISIBLE_DEVICES=6 python main.py --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet_ --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type dice --pw 48 --ph 48 --npatches 120000
#CUDA_VISIBLE_DEVICES=2 python main.py --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type dice
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.001 --workers 2 --epochs 200 --batch-size 128 --checkname unet --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type mIou
#CUDA_VISIBLE_DEVICES=3 python main.py --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type bce
#CUDA_VISIBLE_DEVICES=7 python main.py --backbone resnet --lr 0.007 --workers 2 --epochs 150 --batch-size 128 --checkname deeplab-resnet --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type dice
#============
#Unet
#CUDA_VISIBLE_DEVICES=6 python main.py --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet_ --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type focal --pw 48 --ph 48 --npatches 120000
#CUDA_VISIBLE_DEVICES=7 python main.py --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet0 --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type ce --pw 48 --ph 48 --npatches 120000
#CUDA_VISIBLE_DEVICES=7 python main.py --loss-type mIou --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet0 --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --pw 48 --ph 48 --npatches 120000
#CUDA_VISIBLE_DEVICES=2 python main.py --loss-type focal --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet0 --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --pw 48 --ph 48 --npatches 120000
#CUDA_VISIBLE_DEVICES=1 python main.py --loss-type lovasz_b --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet0 --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --pw 48 --ph 48 --npatches 120000

#==use classweight
#CUDA_VISIBLE_DEVICES=1 python main.py --use-balanced-weights --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet_ --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type ce --pw 48 --ph 48 --npatches 120000

