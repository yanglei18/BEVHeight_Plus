export TORCH_HOME=./$TORCH_HOME
CUDA_VISIBLE_DEVICES=6,7,8,9 python exps/waymo/bev_height_lss_r101_864_1536_256x256.py --amp_backend native -b 2 --gpus 4
CUDA_VISIBLE_DEVICES=6,7,8,9 python exps/waymo/bev_height_lss_r101_864_1536_256x256.py --ckpt outputs/bev_height_lss_r101_864_1536_256x256/checkpoints/ -e -b 2 --gpus 4