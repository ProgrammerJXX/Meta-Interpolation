
# Meta-Interpolation

<!-- [Added] 添加副标题，提升专业度 -->
**An Efficient Seismic Data Interpolation Framework for Adaptive Spatial Continuity Modeling**

<!-- [Added] 添加项目状态徽章和简介 -->
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

This repository provides the official PyTorch implementation for the paper "Meta-Interpolation: An Efficient Seismic Data Interpolation Framework for Adaptive Spatial Continuity Modeling".

## Abstract
Seismic data interpolation is crucial for improving data quality and ensuring reliable subsurface interpretation. While deep learning methods have shown strong potential for this task, they often struggle to preserve the spatial continuity and global consistency of seismic data under complex missing conditions, leading to amplitude distortion and reduced accuracy in subsequent geological interpretation and reservoir evaluation. 

To address this issue, we propose a meta-learning-based framework consisting of an auxiliary network (AN) and an interpolation network (IN), which adaptively learns complete seismic features with strong spatial continuity. The proposed meta-learning framework involves two stages: meta-training and meta-testing. The AN is pre-trained on complete seismic data to capture comprehensive features with strong spatial continuity. During meta-training, the meta-network (MN) is trained to control the distillation loss, enabling the IN to effectively learn seismic prior knowledge from the AN, resulting in a trained MN. During meta-testing, MN adaptively adjusts the distillation loss weights according to the input, guiding IN to learn the seismic prior features from AN, capturing both spatial continuity and global consistency. Extensive experiments conducted under random missing, consecutive missing, and noisy data missing scenarios demonstrate that the proposed framework significantly improves the quality, efficiency, and generalization of seismic interpolation.

## Network Architecture
<!-- [Added] 建议在此处添加框架图，提升可读性 -->
The overall architecture of the proposed framework. The core network `AN_net_Unet` is implemented in `models/compare_models.py`.

<!-- [Added] 图片占位符，如果有图片请取消注释并修改路径 -->
<!-- ![Framework](./figures/framework.png) -->

## Environment Installation
<!-- [Added] 新增环境配置章节，方便他人复现 -->
To run the code, please configure the environment first.

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/Meta-Interpolation.git
    cd Meta-Interpolation
    ```

2. **Create conda environment (Recommended):**
    ```bash
    conda create -n meta_interp python=3.8
    conda activate meta_interp
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies: PyTorch >= 1.10, NumPy, SciPy, Matplotlib.*

## Data Preparation
This project utilizes three distinct public datasets, which have been curated and standardized for research. Please download the data from the official links and place them into the `./data/` directory.

### 1. Dataset SEG C3
*   **Description**: SEGY File Information: 45 shots; 625 samples per trace; 8 ms sample rate; 201 x 201 receiver grid, dx, dy = 20 m.
*   **Download Link**: [SEG C3 Dataset](https://wiki.seg.org/wiki/SEG_C3_45_shot)

### 2. Dataset Model94
*   **Description**: SEGY File Information: 277 shots; 480 traces per shot record; Group interval: 15 m; Shot interval: 90 m.
*   **Download Link**: [Model 94 Dataset](https://wiki.seg.org/wiki/1994_BP_migration_from_topography)

### 3. MAVO Field Dataset
*   **Description**: Mobil AVO Viking Graben Line 12.
*   **Download Link**: [MAVO Dataset](https://wiki.seg.org/wiki/Mobil_AVO_viking_graben_line_12)

## Training
<!-- [Modified] 优化训练命令的展示 -->
We provide an example training script for reproduction. Ensure your data paths are correctly configured in the script.

Run the following command in the terminal:

```bash
sh Meta_interpolation/train_cd_sd_cb_attn_attnd_abl_v4.1_M_30_T_1_L_10_lc_v2_ls_v1_la_v1_beta_1_MAVG_continus_0.1_0.3_AN_net_Unet_4_64_bn_avg_dcn_v2_attn_ca_L1Loss_v2_SSIM_PerceptLoss_pl_num_1_5.sh
```

## Project Structure
<!-- [Added] 新增目录结构说明，帮助他人理解代码组织 -->
The repository is organized as follows:

```
Meta-Interpolation/
├── models/
│   └── compare_models.py    # Implementation of network architectures (AN_net_Unet)
├── Meta_interpolation/      # Main folder for training scripts and utilities
│   └── ...                  # Training shell scripts and core logic
├── data/                    # Directory to store datasets (create manually)
└── README.md
```

## Citation
<!-- [Added] 新增引用格式，这是学术开源代码最重要的一部分 -->
If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{zhang2024meta,
  title={Meta-Interpolation: An Efficient Seismic Data Interpolation Framework for Adaptive Spatial Continuity Modeling},
  author={Zhang, Chunxia and others},
  journal={Computers & Geosciences},
  year={2024}
}
```
*(Note: Please update the year and author list to match your final publication details.)*

## License
<!-- [Added] 新增许可证声明 -->
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
The authors would like to thank the Sandia National Laboratory and Mobil Oil Company for their open data sets.
```

