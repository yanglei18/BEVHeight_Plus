export TORCH_HOME=./$TORCH_HOME
python exps/waymo/bev_height_plus_lss_r101_1024_1536_256x256.py --amp_backend native -b 2 --gpus 8
python exps/waymo/bev_height_plus_lss_r101_1024_1536_256x256.py --ckpt outputs/bev_height_plus_lss_r101_1024_1536_256x256/checkpoints/ -e -b 2 --gpus 8
