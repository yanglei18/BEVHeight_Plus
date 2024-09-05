export TORCH_HOME=./$TORCH_HOME
bash tools/dist_train.sh configs/bevdet/bevdet4d-height-2-r50-depth-cbgs.py 8
bash tools/dist_test.sh configs/bevdet/bevdet4d-height-2-r50-depth-cbgs.py work_dirs/bevdet4d-height-2-r50-depth-cbgs/epoch_8_ema.pth 8 --eval mAP