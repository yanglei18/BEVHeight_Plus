<p align="center">

  <h1 align="center">BEVHeight++: Toward Robust Visual Centric 3D Object Detection</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=EUnI2nMAAAAJ&hl=zh-CN"><strong>Lei Yang</strong></a>
    · 
    <a href="https://scholar.google.com.hk/citations?user=1ltylFwAAAAJ&hl=zh-CN&oi=sra"><strong>Tao Tang</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong>Jun Li</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=aMnHLz4AAAAJ&hl=zh-CN&oi=ao"><strong>Kun Yuan</strong></a>
    ·
    <a href="https://damo.alibaba.com/labs/intelligent-transportation"><strong>Peng Chen</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=kLTnwAsAAAAJ&hl=zh-CN&oi=sra"><strong>Li Wang</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong>Yi Huang</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=KId65yQAAAAJ&hl=zh-CN&oi=ao"><strong>Lei Li</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=0Q7pN4cAAAAJ&hl=zh-CN&oi=sra"><strong>Xinyu Zhang</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=Jtmq_m0AAAAJ&hl=zh-CN&oi=sra"><strong>Kaicheng Yu</strong></a>
  </p>


<h2 align="center"></h2>
  <div align="center">
    <img src="../assets/BEVHeight++.jpg" alt="Logo" width="88%">
  </div>

<p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
     <a href='https://hub.docker.com/repository/docker/yanglei2024/op-bevheight/general'><img src='https://img.shields.io/badge/Docker-9cf.svg?logo=Docker' alt='Docker'></a>
    <br></br>
    </a>
  </p>
</p>

**BEVHeight++** is a new vision-based 3D object detector specially designed for both roadside and vihicle-side scenarios. On popular 3D detection benchmarks of roadside cameras, BEVHeight++ surpasses all previous vision-centric methods by a significant margin. In terms of the ego-vehicle scenario, our BEVHeight++ also possesses superior over depth-only methods.

# Getting Started

 ### 0. Installation as [BEVDet](https://github.com/HuangJunJie2017/BEVDet)


 ### 1. Prepare nuScenes Dataset
 Download nuScenes dataset from official [website](https://www.nuscenes.org/nuscenes).

```shell
ln -s [nuScenes-dataset-root] ./data/nuscenes
```

### 2. Prepare infos for nuScenes dataset.
```shell
python tools/create_data_bevheight_plus.py
```

### 3. Train BEVHeight++ with 8 GPUs
```shell
# stage 1: 
bash tools/dist_train.sh configs/bevheight_plus/bevheight_plus-r50-depth-cbgs-first-stage.py 8

# stage 2:
mv [PTH_PATH_OF_STAGE_1] pretrained_model/epoch_20_ema.pth
bash tools/dist_train.sh configs/bevheight_plus/bevheight_plus-r50-depth-cbgs.py 8
```

### 4. Eval BEVHeight++ with 8 GPUs
```shell
bash tools/dist_test.sh configs/bevheight_plus/bevheight_plus-r50-depth-cbgs.py [PTH_PATH_OF_STAGE_2] 8 --eval mAP
```

# Results on NuScenes Test Set.
| Model | Backbone| mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS | Config | Download|
| :---: | :---: | :---: | :---:|:---:| :---: | :---:| :---: | :---: | :---: | :---: |
|BEVHeight++| V2-99 | 0.529 | 0.441 | 0.258 | 0.358 | 0.295 | 0.142 | 0.614 |[config](configs/bevheight_plus/bevheight_plus-vov-depth-cbgs-900-1600.py) |[model](https://cloud.tsinghua.edu.cn/f/7251e41c74e74c97ada5/?dl=1) /  [log](https://cloud.tsinghua.edu.cn/f/cefb37bbb53c40118279/?dl=1) |

# Acknowledgment
This project is not possible without the following codebases.
* [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)

# Citation
If you use BEVHeight++ in your research, please cite our work by using the following BibTeX entry:
```
@article{yang2023bevheight++,
  title={Bevheight++: Toward robust visual centric 3d object detection},
  author={Yang, Lei and Tang, Tao and Li, Jun and Chen, Peng and Yuan, Kun and Wang, Li and Huang, Yi and Zhang, Xinyu and Yu, Kaicheng},
  journal={arXiv preprint arXiv:2309.16179},
  year={2023}
}
```
