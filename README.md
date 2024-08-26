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
    <img src="./assets/BEVHeight++.jpg" alt="Logo" width="88%">
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

# News

- [2024/06/30] Both arXiv and codebase are released!
# Incoming

- [ ] Release the pretrained models

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Getting Started">Getting Started</a>
    </li>
    <li>
      <a href="#Acknowledgment">Acknowledgment</a>
    </li>
    <li>
      <a href="#Citation">Citation</a>
    </li>
  </ol>
</details>

<br/>

# Getting Started

- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)

Train BEVHeight with 8 GPUs
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
Eval BEVHeight with 8 GPUs
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```

# Experimental Results
- **KITTI Dataset**
<div align=left>
<table>
     <tr align=center>
        <td rowspan="3">Method</td> 
        <td colspan="3" align=center>AP|3D</td>
        <td colspan="3" align=center>AP|BEV</td>
        <td rowspan="3" align=center>config</td> 
        <td rowspan="3" align=center>model pth</td>
    </tr>
    <tr align=center>
    </tr>
    <tr align=center>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
    </tr>
    <tr align=center>
        <td>BEVDepth</td> 
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight</td> 
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight++</td> 
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    
<table>
</div>

- **KITTI-360 Dataset**
<div align=left>
<table>
     <tr align=center>
        <td rowspan="3">Method</td> 
        <td colspan="3" align=center>AP3D (IoU=0.5)</td>
        <td colspan="3" align=center>AP3D (IoU=0.25)</td>
        <td rowspan="3" align=center>config</td> 
        <td rowspan="3" align=center>model pth</td>
    </tr>
    <tr align=center>
    </tr>
    <tr align=center>
        <td>AP (Lrg)</td>
        <td>AP (Car)</td>
        <td>mAP</td>
        <td>AP (Lrg)</td>
        <td>AP (Car)</td>
        <td>mAP</td>
    </tr>
    <tr align=center>
        <td>BEVDepth</td> 
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight</td> 
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight++</td> 
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    
<table>
</div>

- **Waymo Dataset**
<div align=left>
<table>
     <tr align=center>
        <td rowspan="3">IoU3D</td> 
        <td rowspan="3">Difficulty</td> 
        <td rowspan="3">Method</td> 
        <td colspan="4" align=center>AP3D</td>
        <td colspan="4" align=center>APH3D</td>
        <td rowspan="3" align=center>config</td> 
        <td rowspan="3" align=center>model pth</td>
    </tr>
    <tr align=center>
    </tr>
    <tr align=center>
        <td>All</td>
        <td>0-30</td>
        <td>30-50</td>
        <td>50- OO </td>
        <td>All</td>
        <td>0-30</td>
        <td>30-50</td>
        <td>50-OO</td>
    </tr>
    <tr align=center>
        <td rowspan="3">0.7</td> 
        <td rowspan="3">Level_1</td> 
        <td>BEVDepth</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight++</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td rowspan="3">0.7</td> 
        <td rowspan="3">Level_2</td> 
        <td>BEVDepth</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight++</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td rowspan="3">0.5</td> 
        <td rowspan="3">Level_1</td> 
        <td>BEVDepth</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight++</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td rowspan="3">0.5</td> 
        <td rowspan="3">Level_2</td> 
        <td>BEVDepth</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td>BEVHeight++</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr> 
<table>
</div>

# Acknowledgment
This project is not possible without the following codebases.
* [BEVHeight](https://github.com/ADLab-AutoDrive/BEVHeight)
* [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
* [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
* [pypcd](https://github.com/dimatura/pypcd)

# Citation
If you use BEVHeight in your research, please cite our work by using the following BibTeX entry:
```
@article{yang2023bevheight++,
  title={Bevheight++: Toward robust visual centric 3d object detection},
  author={Yang, Lei and Tang, Tao and Li, Jun and Chen, Peng and Yuan, Kun and Wang, Li and Huang, Yi and Zhang, Xinyu and Yu, Kaicheng},
  journal={arXiv preprint arXiv:2309.16179},
  year={2023}
}
```
