GPUS=1
NNODES=1
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

# install reqs
bash setup_local.sh

torchrun \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS \
    bev_blip2_laion_multinode.py \
    --laion_tar_files "/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/{00100..00200}.tar" \
    --out_path "/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v2-prelim/" \
    --timestamp_out_path True \
    --render_selected True \
    --render_orig True
