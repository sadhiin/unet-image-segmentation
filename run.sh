# ## -------------- For Experiment with KITTI Dataset --------------
# ## if the dataset is downloaded, then you can replace the data_dir with the path to the dataset as follows:
# ## --data
# ##      |-kitti-dataset
# ##          |-all kitti data

# python train.py \
#     --dataset kitti \
#     --data_dir data \
#     --image_size 256 \
#     --batch_size 16 \
#     --epochs 2 \
#     --learning_rate 0.00001 \
#     --num_workers 2 \
#     --use_wandb

# ## -------------- For Experiment with CityScapes Dataset --------------
# ## if the dataset is downloaded, then you can replace the data_dir with the path to the dataset as follows:
# ## -- data
# ##      |-cityscapes_data
# ##          |-all cityscapes data
# python train.py \
#     --dataset cityscapes \
#     --data_dir data \
#     --image_size 256 \
#     --batch_size 16 \
#     --epochs 2 \
#     --learning_rate 0.00001 \
#     --num_workers 2 \
#     --use_wandb

# ## -------------- For Experiment with Pascal VOC Dataset --------------
# ## if the dataset is downloaded, then you can replace the data_dir with the path to the dataset as follows:
# ## -- data
# ##        |-pascal_voc
# ##            |-....
# python train.py \
#     --dataset pascal_voc \
#     --data_dir data \
#     --image_size 256 \
#     --batch_size 16 \
#     --epochs 5 \
#     --learning_rate 0.0001 \
#     --num_workers 2 \
#     --use_wandb

# ## -------------- For Experiment with COCO Dataset --------------
# ## if the dataset is downloaded, then you can replace the data_dir with the path to the dataset as follows:
# ## -- data
# ##        |-coco
# ##            |-....
python train.py \
    --dataset coco \
    --data_dir data \
    --image_size 256 \
    --batch_size 16 \
    --epochs 2 \
    --learning_rate 0.0001 \
    --num_workers 2 \
    --use_wandb