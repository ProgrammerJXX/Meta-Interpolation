# Abstract
Seismic data interpolation is crucial for improving data quality and ensuring reliable subsurface interpretation. While deep learning methods have shown strong potential for this task, they often struggle to preserve the spatial continuity and global consistency of seismic data under complex missing conditions, leading to amplitude distortion and reduced accuracy in subsequent geological interpretation and reservoir evaluation. To address this issue, we propose a meta-learning-based framework consisting of an auxiliary network (AN) and an interpolation network (IN), which adaptively learns complete seismic features with strong spatial continuity. The proposed meta-learning framework involves two stages: meta-training and meta-testing. The AN is pre-trained on complete seismic data to capture comprehensive features with strong spatial continuity. During meta-training, the meta-network (MN) is trained to control the distillation loss, enabling the IN to effectively learn seismic prior knowledge from the AN, resulting in a trained MN. During meta-testing, MN adaptively adjusts the distillation loss weights according to the input, guiding IN to learn the seismic prior features from AN, capturing both spatial continuity and global consistency. Extensive experiments conducted under random missing, consecutive missing, and noisy data missing scenarios demonstrate that the proposed framework significantly improves the quality, efficiency, and generalization of seismic interpolation.

# Meta-Interpolation
This project is about Meta-Interpolation: An Efficient Seismic Data Interpolation Framework for Adaptive Spatial Continuity Modeling.The code will be made public in the future.

# Public Dataset Release
This page announces the public release of three distinct datasets, curated and standardized for research and reuse.
## Dataset SEG C3
Description: [SEGY File Information:45 shots; 625 samples per trace; 8 ms sample rate; 201 x 201 receiver grid, dx,dy = 20 m]

Data Link: [download link for SEG C3](https://wiki.seg.org/wiki/SEG_C3_45_shot)


## Dataset Model94
Description: [SEGY File Information:Number of shots: 277 shots;Number of traces per shot: 480 traces per shot record;Group interval: 15 m;Shot interval: 90 m]

Data Link: [download link for Model 94]([https://wiki.seg.org/wiki/SEG_C3_45_shot](https://wiki.seg.org/wiki/1994_BP_migration_from_topography))

## MAVO Field Dataset
Description: [SEGY File Information:]

Data Link: [download link for MAVO]([[https://wiki.seg.org/wiki/SEG_C3_45_shot](https://wiki.seg.org/wiki/1994_BP_migration_from_topography](https://wiki.seg.org/wiki/Mobil_AVO_viking_graben_line_12)))

## Training
We have provided the example training scripts. Run in terminal:
```
 sh Meta_interpolation/train_cd_sd_cb_attn_attnd_abl_v4.1_M_30_T_1_L_10_lc_v2_ls_v1_la_v1_beta_1_MAVG_continus_0.1_0.3_AN_net_Unet_4_64_bn_avg_dcn_v2_attn_ca_L1Loss_v2_SSIM_PerceptLoss_pl_num_1_5.sh
```
