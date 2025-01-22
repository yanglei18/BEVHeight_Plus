## Installation
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch

pip install mmcv-full==1.5.2  mmdet==2.24.0 mmsegmentation==0.29.0
pip install nuscenes-devkit==1.1.10
pip install numba==0.53.0 numpy==1.21.2

cd BEVHeight_Plus/nuscense_bevheight_plus
python setup.py build develop
```
## Data Prepare

>* Download nuScenes Dataset
>* Downlowd [`bevheight_plus_nuscenes_infos_all.pkl`](https://cloud.tsinghua.edu.cn/f/18803f287e2a4f20a1a9/?dl=1) and [`bevheight_plus_nuscenes_infos_val.pkl`](https://cloud.tsinghua.edu.cn/f/bee57ee6237f42cca3fa/?dl=1), [`bevheight_plus_nuscenes_infos_test.pkl`](https://cloud.tsinghua.edu.cn/f/0b84d3ea57d24e18b7ce/?dl=1)
>* Download  [`fcos3d_vovnet_imgbackbone-remapped.pth`](https://cloud.tsinghua.edu.cn/f/df9230cea69e4eeab116/?dl=1)

```shell
cd BEVHeight_Plus/nuscense_bevheight_plus 
mkdir data && mkdir pretrained_model
ln -s [nuScenes-dataset-root] ./data/nuscenes
mv bevheight_plus_nuscenes_infos_all.pkl ./data/nuscenes
mv bevheight_plus_nuscenes_infos_val.pkl ./data/nuscenes
mv bevheight_plus_nuscenes_infos_test.pkl ./data/nuscenes
mv fcos3d_vovnet_imgbackbone-remapped.pth ./pretrained_model
```

## Train
```shell
bash tools/dist_train.sh configs/bevheight_plus/bevheight_plus-vov-depth-cbgs-900-1600.py 8
```

## Evaluation
```shell
bash tools/dist_test.sh configs/bevheight_plus/bevheight_plus-vov-depth-cbgs-900-1600.py [PTH_PATH] 8 --eval mAP
```


