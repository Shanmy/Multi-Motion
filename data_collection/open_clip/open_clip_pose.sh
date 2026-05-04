cd src
torchrun --nproc_per_node 8 -m training.main_pose \
    --train-data '/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/yutao_text_rewrite/{00701..00799}.tar' \
    --train-num-samples 10968539 \
    --epochs 400 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --dataset-resampled \
    --logs /nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/pose/ \
