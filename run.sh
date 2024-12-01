## -------------- For Experiment with KITTI Dataset --------------
# python train.py \
#     --dataset kitti \
#     --data_dir /teamspace/studios/this_studio/unet-for-segmentation/data \
#     --image_size 256 \
#     --batch_size 16 \
#     --epochs 2 \
#     --learning_rate 0.00001 \
#     --num_workers 2 \
#     --use_wandb

## -------------- For Experiment with CityScapes Dataset --------------
# python train.py \
#     --dataset cityscapes \
#     --data_dir /teamspace/studios/this_studio/unet-for-segmentation/data \
#     --image_size 256 \
#     --batch_size 16 \
#     --epochs 2 \
#     --learning_rate 0.00001 \
#     --num_workers 2 \
#     --use_wandb

## -------------- For Experiment with Pascal VOC Dataset --------------
python train.py \
    --dataset pascal_voc \
    --data_dir /teamspace/studios/this_studio/unet-for-segmentation/data \
    --image_size 256 \
    --batch_size 16 \
    --epochs 5 \
    --learning_rate 0.0001 \
    --num_workers 2 \
    --use_wandb