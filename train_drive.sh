#CUDA_VISIBLE_DEVICES=0 python main.py --model_name deeplab --backbone resnet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname deeplab-resnet --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type ce --pw 96 --ph 96 --npatches 120000 --inchannels 1
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
#CUDA_VISIBLE_DEVICES=2 python main.py --loss-type ohem --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname ohem --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --pw 48 --ph 48 --npatches 120000
#CUDA_VISIBLE_DEVICES=4 python main.py --loss-type ge-dice --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet0 --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --pw 48 --ph 48 --npatches 120000

#two loss
#CUDA_VISIBLE_DEVICES=3 python main.py --loss-type ce_dice --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname sum_loss --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --pw 48 --ph 48 --npatches 120000

#==use classweight
#CUDA_VISIBLE_DEVICES=1 python main.py --use-balanced-weights --model_name unet --lr 0.01 --workers 2 --epochs 200 --batch-size 128 --checkname unet_ --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --loss-type ce --pw 48 --ph 48 --npatches 120000

#==========focal loss
CUDA_VISIBLE_DEVICES=4 python main.py --loss-type focal --model_name unet --lr 0.01 --workers 2 --epochs 120 --batch-size 128 --checkname focal --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --pw 48 --ph 48 --npatches 12000


#=======triplet
#CUDA_VISIBLE_DEVICES=0 python main.py --loss-type tri --model_name unet_tri --lr 0.01 --workers 2 --epochs 120 --batch-size 16 --checkname unet-tri --eval-interval 1 --dataset drive --sync-bn False --gpu-ids 0 --pw 48 --ph 48 --npatches 120000
