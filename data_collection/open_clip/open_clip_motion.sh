cd src
torchrun --nproc_per_node 4 -m training.main_pose \
    --train-num-samples 15000 \
    --epochs 100 \
    --dataset-type amass \
    --batch-size 128 \
    --precision amp \
    --workers 4 \
    --dataset-resampled \
    --logs /nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/pose/ \
