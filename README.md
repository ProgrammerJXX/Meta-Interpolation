
# Meta-Interpolation: An Efficient Seismic Data Interpolation Framework for Adaptive Spatial Continuity Modeling


## 🔭 Project Overview

**Meta-Interpolation** is a novel meta-learning-based framework for seismic data interpolation that adaptively learns complete seismic features with strong spatial continuity. Unlike conventional deep learning methods that struggle to preserve spatial continuity under complex missing conditions, our approach introduces a dual-network architecture with adaptive distillation loss to maintain both local and global consistency in reconstructed seismic data.

**Key Contributions:**
- **Meta-learning framework** with auxiliary network (AN) and interpolation network (IN)
- **Adaptive spatial continuity modeling** through meta-network controlled distillation
- **Better performance** under random missing, consecutive missing, and noisy scenarios
- **Enhanced generalization** across different geological settings and acquisition geometries


## 📁 Code Architecture

The project is organized as follows:


| Directory/File | Description |
|----------------|-------------|
| `data/` | Scripts for data preprocessing, augmentation, and missing pattern generation |
| `dataset/` | PyTorch Dataset classes for loading different seismic datasets |
| `models/` | Implementation of all network architectures (AN, IN, MN) and loss functions |
| `utils/` | Helper functions for metrics, visualization, logging, and checkpointing |
| `train_cd_sd_cb_attn_attnd.py` | Main entry point for training the meta-learning framework |
| `requirements.txt` | Python package dependencies |

<!-- 添加：代码架构详细说明 -->

### 🏗 Network Architecture

Our main interpolation network `IN_net_Unet` is implemented in `models/compare_models.py`.

### Network Components

| Network | File | Description |
|---------|------|-------------|
| **Auxiliary Network (AN)** | `models/compare_models.py` | Pre-trained on complete seismic data |
| **Interpolation Network (IN)** | `models/compare_models.py` | Main network for reconstruction |
| **Meta-Network (MN)** | `models/meta_network.py` | Controls distillation loss adaptation |
| **Loss Functions** | `utils/loss.py` | Combined L1, SSIM, Perceptual losses |

<!-- 添加：网络架构详细说明，包含文件位置 -->

### 📊 Datasets

We provide three benchmark datasets for seismic interpolation research:
<img width="1530" height="184" alt="image" src="https://github.com/user-attachments/assets/5d535d69-273e-4d62-a54b-9fd59aa83ec0" />


| Attribute | SEG C3 | Model94 | MAVO Field |
|-----------|--------|---------|------------|
| **Download** | [Link](https://wiki.seg.org/wiki/SEG_C3_45_shot) | [Link](https://wiki.seg.org/wiki/1994_BP_migration_from_topography) | [Link](https://wiki.seg.org/wiki/Mobil_AVO_viking_graben_line_12) |

## 💻 Installation

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Meta-Interpolation.git
cd Meta-Interpolation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📁 Data Preparation

### 1. Download Raw Data
Download the SEG-Y files from the provided links in the [Datasets](#datasets) section.

### 2. Preprocess Data
Use the preprocessing scripts in the `data/` directory:

```bash
python data/SEGC3.py 
```

### Expected Directory Structure After Preparation
```
Meta_interpolation/
├── data/
│   ├── seg_c3/
│   ├── model94/
│   └── mavo/
```

<!-- 添加：详细的数据准备步骤，包含具体的脚本调用 -->

## 🚀 Training

### Main Training Script

The main entry point for training is `train_cd_sd_cb_attn_attnd.py`:

```bash
python train_cd_sd_cb_attn_attnd.py
```

### Example Script

```bash
sh Meta_interpolation/train_cd_sd_cb_attn_attnd_abl_v4.1_M_30_T_1_L_10_lc_v2_ls_v1_la_v1_beta_1_MAVG_continus_0.1_0.3_AN_net_Unet_4_64_bn_avg_dcn_v2_attn_ca_L1Loss_v2_SSIM_PerceptLoss_pl_num_1_5.sh
```

### Visualization

```bash
python utils/plot.py 
```

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{Jiang etal.  title={Meta-Interpolation: An Efficient Seismic Data Interpolation Framework for Adaptive Spatial Continuity Modeling}
}
```


## 🙏 Acknowledgments

- SEG for providing open-access seismic datasets
- [SEG Wiki](https://wiki.seg.org) for dataset hosting
- Contributors and reviewers who provided valuable feedback

## 📬 Contact

For questions or collaboration opportunities:
- **GitHub Issues**: [Open an issue](https://github.com/yourusername/Meta-Interpolation/issues)


