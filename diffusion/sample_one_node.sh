GPUS=${1:-8}
CONFIG=${2:-"configs_sample/pose2motion_2_stage_webvid_frozen.py"}  # Edit paths in this config before running
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"

torchrun \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS \
    base_sample.py \
    $CONFIG \
    --MASTER $MASTER_ADDR
