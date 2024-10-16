export TORCH_HOME=./$TORCH_HOME
python exps/kitti/bev_depth_lss_r101_384_1280_256x256.py --amp_backend native -b 2 --gpus 8
python exps/kitti/bev_depth_lss_r101_384_1280_256x256.py --ckpt outputs/bev_depth_lss_r101_384_1280_256x256/checkpoints/ -e -b 2 --gpus 8
