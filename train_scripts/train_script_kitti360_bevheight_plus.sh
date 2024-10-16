export TORCH_HOME=./$TORCH_HOME
python exps/kitti-360/bev_height_plus_lss_r101_384_1280_256x256.py --amp_backend native -b 2 --gpus 8
python exps/kitti-360/bev_height_plus_lss_r101_384_1280_256x256.py --ckpt outputs/bev_height_plus_lss_r101_384_1280_256x256/checkpoints/ -e -b 2 --gpus 8
