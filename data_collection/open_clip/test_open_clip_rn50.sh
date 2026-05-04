cd src
torchrun --nproc_per_node 8 -m training.main \
    --train-data '/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/{00000..32449}.tar' \
    --train-num-samples 10968539 \
    --epochs 400 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /nfs/SHARE/dataset/imagenet/val/ \
    --dataset-resampled \
    --logs /nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/image_validation/ \
