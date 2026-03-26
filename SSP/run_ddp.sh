#!/bin/bash
# DDP 训练启动脚本
# 使用方法:
#   bash run_ddp.sh [GPU数量] [SSP|SP]
# 例如:
#   bash run_ddp.sh 2 SSP   # 默认：SSP_ESM3_train_ddp.py
#   bash run_ddp.sh 2 SP    # 单模型：SP_ESM3_train_ddp.py
# 单卡运行请直接：python <脚本名>.py

NUM_GPUS=${1:-2}
MODE=${2:-SSP}

if [[ "${MODE}" == "SP" ]]; then
  TRAIN_SCRIPT="SP_ESM3_train_ddp.py"
else
  TRAIN_SCRIPT="SSP_ESM3_train_ddp.py"
fi

echo "Starting DDP training with ${NUM_GPUS} GPUs... mode=${MODE} script=${TRAIN_SCRIPT}"

# 设置 CUDA 可见设备（根据 GPU 数量自动生成）
CUDA_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
# export CUDA_LAUNCH_BLOCKING=1

# 关键修复：禁用 NCCL P2P 和 SHM（RTX PRO 6000 Blackwell 需要）
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_DEBUG=WARN

# 使用 torchrun 启动
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    ${TRAIN_SCRIPT}

# 或者使用旧版 launch 方式（如果 torchrun 不可用）
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --use_env \
#     SSP_ESM3_train_ddp.py

