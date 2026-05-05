GPUS=${1:-4}
NNODES=${2:-1}
CONFIG=${3:-"configs_train/pose2motion_2_stage.py"}  # Edit paths in this config before running

NODE_RANK=${RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NCCL_IB_HCA=`ibdev2netdev|awk '{print$1}'`
roce_PORT=":1"
NCCL_IB_HCA=${NCCL_IB_HCA}${roce_PORT}
NCCL_DEBUG=TRACE
OMPI_MCA_btl_tcp_if_include=eth0
NCCL_SOCKET_IFNAME=eth0
NCCL_IB_DISABLE=0
NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA
export NCCL_DEBUG
export OMPI_MCA_btl_tcp_if_include
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE
export NCCL_IB_GID_INDEX
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
