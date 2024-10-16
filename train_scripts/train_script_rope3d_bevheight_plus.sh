export TORCH_HOME=./$TORCH_HOME
python exps/rope3d/bev_height_plus_lss_r101_864_1536_256x256_102.py --amp_backend native -b 2 --gpus 8
python exps/rope3d/bev_height_plus_lss_r101_864_1536_256x256_102.py --ckpt outputs/bev_height_plus_lss_r101_864_1536_256x256_102/checkpoints/ -e -b 2 --gpus 8
