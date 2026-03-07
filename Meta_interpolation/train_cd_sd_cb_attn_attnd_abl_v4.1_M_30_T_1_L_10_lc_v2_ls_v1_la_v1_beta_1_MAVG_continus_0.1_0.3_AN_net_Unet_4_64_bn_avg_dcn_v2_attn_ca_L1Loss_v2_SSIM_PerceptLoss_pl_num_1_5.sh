#!/D:/Git/bin/bash

SCRIPT_DIR=$(dirname "$0")
sh_name=$(basename "$0" .sh) # $(basename "$0")

# 检测操作系统类型
OS_TYPE="$(uname)"  # 获取操作系统名称

if [ "$OS_TYPE" = "Linux" ]; then
    # Linux 系统
    UBUNTU_HOSTNAME=$(hostname)
    case $UBUNTU_HOSTNAME in
        "ubuntu")
        python_exec="python"
        script_path="/home/code/Distill_model/train_cd_sd_cb_attn_attnd.py"
        GPU=1
            ;;
        *)
            echo "Unrecognized hostname"
            exit 1
            ;;
    esac
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi


# Lrs列表
Lrs=("5e-3")
# 循环遍历数据集列表
for lr in "${Lrs[@]}"; do
    echo "==============================================="
    echo " 开始训练 lr: $lr "
    echo "==============================================="
    current_time=$(date "+%Y-%m-%d-%H-%M-%S")

    # 执行 Python 脚本
    nohup "$python_exec" -u -W ignore "$script_path" \
    --mode train \
    --Epoch 120 \
    --gpu $GPU \
    --model_source_load_path Distill_model/results/train_source_MAVG_AN_net_Unet_4_64_bn_avg_attn_spa_L1Loss_v1/2025-07-23-21-15-37/checkpt.pth \
    --print_interval 50 \
    --save_interval 100 \
    --online_vis swanlab \
    --dataset MAVG \
    --data_use 0.3 \
    --missing_p 0.3 \
    --seed 123 \
    --noise_level_img 0 \
    --meta_batch 32 \
    --batch_train 32 \
    --batch_val 32 \
    --batch_test 32 \
    --resolution_h 256 \
    --resolution_w 112 \
    --sampler continus \
    --num_workers 2 \
    --input_dim 1 \
    --depth 4 \
    --cnum 64 \
    --norm bn \
    --pool avg \
    --activation relu \
    --optimizer ADAM \
    --lr $lr \
    --wd 1e-4 \
    --betas 0.9 0.999 \
    --scheduler StepLR \
    --num_workers 2 \
    --step_size 100 \
    --gamma 0.1 \
    --momentum 0.9 \
    --distill_type v4.1 \
    --need_cd \
    --need_sd \
    --source_model AN_net_Unet \
    --target_model IN_net_Unet \
    --source_model_sd IN_net_Unet_deep \
    --target_model_sd IN_net_Unet_shallow \
    --source_model_attnd AN_net_Unet_attn \
    --target_model_attnd IN_net_Unet_attn \
    --pairs 0-0 \
    --pairs_sd 0-0,1-1 \
    --max_meta_step 30 \
    --T 1 \
    --L 10 \
    --roll_back \
    --feature_matching_type v2 \
    --feature_matching_type_sd v1 \
    --feature_matching_type_attnd v1 \
    --meta_lr 5e-4 \
    --meta_wd 1e-4 \
    --loss_weight_cd \
    --loss_weight_sd \
    --loss_weight_type relu \
    --loss_weight_init 1 \
    --loss_weight_init_sd 1 \
    --loss_weight_init_attnd 1 \
    --beta 1 \
    --beta_mt 1 \
    --use_mask \
    --loss_list L1Loss SSIMLoss PerceptLoss \
    --loss_weight 1 1 0.1 \
    --l1_type v2 \
    --deform_conv_type v2 \
    --use_deform_layer_down 100 \
    --use_deform_layer_up 100 \
    --use_attention \
    --attention_type ca \
    --use_attn_layer_down 100 \
    --use_attn_layer_up 0 1 2 \
    --num_heads 1 \
    --reduction_ratio 8 \
    --train_data_dir Hint_seismic/dataset/mavg/train \
    --val_data_dir Hint_seismic/dataset/mavg/val \
    --test_data_dir Hint_seismic/dataset/mavg/test \
    --save Distill_model/results/$sh_name/$current_time \
    --pt_path Distill_model/results/$sh_name/$current_time/checkpt.pth \
    2>&1 | tee $SCRIPT_DIR/log_${sh_name}_${current_time}.txt
    echo "lr $L 训练完成！"
    echo "-----------------------------------------------------------------------------------------------"
done
