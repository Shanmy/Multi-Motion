GPUS=${1:-8}
CONFIG=${2:-"configs_train/pose2motion_2_stage.py"}  # Edit paths in this config before running
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
PORT=${PORT:-29500}

export PYTHONPATH=..:$PYTHONPATH

torchrun \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS \
    base_train.py \
    $CONFIG \
    --MASTER $MASTER_ADDR
