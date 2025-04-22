# CONFIG=/datasata/xuronghao/coding/Diffusion-Low-Light-main/configs/SIG17_hdr.yml
# RESULT_IMAGE_FOLDER=/datasata/xuronghao/coding/Diffusion-Low-Light-main/HDRresults

NUM_GPUS=$1
CUDA_DEVICES=$2
CONFIG=$3
# python -m torch.distributed.launch
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export OMP_NUM_THREADS=4

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# --master_port=25461
torchrun --nproc_per_node=$NUM_GPUS $PROJECT_ROOT/train.py \
    --config $CONFIG \
    --seed 2025 \
    --sampling_timesteps 10 