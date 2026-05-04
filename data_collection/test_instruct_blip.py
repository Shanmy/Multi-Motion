import os
from datetime import datetime
import argparse

import numpy as np
from PIL import Image

import torch

from net_utils import import_instruct_blip
from data_laion import load_img_tar
from data_pose import LaionPose
from viz_utils import viz_render

parser = argparse.ArgumentParser()
parser.add_argument("--pose_tar_files", type=str, default="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v2-prelim/23-10-10-01-50-55/{00000..00007}.tar")
parser.add_argument("--laion_path", type=str, default="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/")
parser.add_argument("--burnin_batches", type=int, default=100)
parser.add_argument("--out_path", type=str, default="out_laion_viz/")
parser.add_argument("--num_samples_viz", type=int, default=50)
parser.add_argument("--num_sample_reps", type=int, default=4, help="number of replicates for model input (use to test memory limitations)")
parser.add_argument("--prompt", type=str, default="Describe the action and body position of the person or people.")
args = parser.parse_args()

# folder to save viz results
out_path = os.path.join(args.out_path, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
os.makedirs(out_path, exist_ok=True)

# load net and data processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = import_instruct_blip(device)

# make dataloader for wds pose data
pose_dataset = LaionPose(urls=args.pose_tar_files)
pose_dataloader = pose_dataset.wds_loader(
                        batch_size=args.num_samples_viz,
                        resampled=True,
                        yield_instruct_blip_text=False,
                        yield_beta=True,
                        yield_key=True,
                    )
# burin samples to shuffle better
pose_iterator = iter(pose_dataloader)
for _ in range(args.burnin_batches):
    _ = next(pose_iterator)

# get batch for viz
pose_samples = next(iter(pose_dataloader))
thetas_samples = pose_samples[0]
trans_samples = pose_samples[1]
betas_samples = pose_samples[4]
key = pose_samples[5]
sample_iterator = zip(
    betas_samples,
    thetas_samples,
    trans_samples,
    key
)

for i, (betas, thetas, trans, key) in enumerate(sample_iterator):
    print(f'Visualizing sample {i+1}.')
    orig_image = load_img_tar(key, laion_path=args.laion_path)
    orig_image = torch.tensor(np.stack(args.num_sample_reps * [orig_image])).permute(0, 3, 1, 2).cuda()
    prompt = args.num_sample_reps * [args.prompt]

    inputs = processor(images=orig_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=50,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    viz_render(
        thetas,
        trans,
        generated_text,
        key,
        betas=betas,
        out_path=out_path,
        laion_path=args.laion_path,
    )
