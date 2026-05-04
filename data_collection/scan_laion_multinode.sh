GPUS=${1:-1}
NNODES=${2:-1}
USER_DIR=${3:-"us000240"}

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


python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS \
    scan_laion.py \
    --MASTER $MASTER_ADDR \
