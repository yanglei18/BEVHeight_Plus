export TORCH_HOME=./$TORCH_HOME
python exps/dair-v2x/bev_depth_lss_r101_864_1536_256x256_102.py --amp_backend native -b 2 --gpus 8
python exps/dair-v2x/bev_depth_lss_r101_864_1536_256x256_102.py --ckpt outputs/bev_depth_lss_r101_864_1536_256x256_102/checkpoints/ -e -b 2 --gpus 8
