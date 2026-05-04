cd src
torchrun --nproc_per_node 1 -m training.main \
    --imagenet-val /nfs/SHARE/dataset/imagenet/val/ \
    --model ViT-H-14 \
    --pretrained /nfs/USRCSEA/IVA/Models/PoseCaptionData/open_clip/vit_h_14_open_clip_pytorch_model.bin \
    --logs /nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/image_validation_eval/ \
