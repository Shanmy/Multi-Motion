import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm


def get_eval_dict(
    folder,
    metrics=(
        "CLPP FID",
        "CLPP Sim",
        "CLPP Diversity",
        "CLPP R Prec Top 1",
        "CLPP R Prec Top 2",
        "CLPP R Prec Top 3",
        "CLMP FID",
        "CLMP Sim",
        "CLMP Diversity",
        "CLMP R Prec Top 1",
        "CLMP R Prec Top 2",
        "CLMP R Prec Top 3",
    ),
):
    eval_dict = {}
    for file in os.listdir(folder):
        if file.startswith('eval'):
            with open(os.path.join(folder, file), 'r') as f:
                text_lines = f.readlines()
            for line in text_lines:
                for metric in metrics:
                    if line.startswith(metric + ":"):
                        line.replace("\n", "")
                        eval_dict[metric] = float(line.split(" ")[-1])
            break
    return eval_dict

test_folder = '/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-restart-v1/pose2motion_2_stage_webvid_freeze_pose_8_gpu/24-03-01-21-59-52/samples/pose2motion_2_stage_4_gpu/24-03-03-06-12-45'
subfolders = os.listdir(test_folder)

pose_coefs = []
motion_coefs = []
eval_dicts = []

for folder in subfolders:
    if folder.startswith('pose'):
        pose_coefs.append(float(folder.split("_")[1]))
        motion_coefs.append(float(folder.split("_")[3]))
        eval_dict = get_eval_dict(os.path.join(test_folder, folder))
        if not eval_dict == {}:
            eval_dicts.append(eval_dict)

for metric in eval_dicts[0].keys():
    # get high and low values for each metric to scale viz colors
    high_val = -1e99
    low_val = 1e99
    for eval_dict in eval_dicts:
        metric_val = eval_dict[metric]
        if metric_val > high_val:
            high_val = metric_val
        if metric_val < low_val:
            low_val = metric_val

    metric_vals = []
    for eval_dict in eval_dicts:
        metric_val = eval_dict[metric]
        metric_rescale = (metric_val - low_val) / (high_val - low_val + 1e-10)
        metric_vals.append(metric_rescale)
    metric_vals = tuple(metric_vals)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(pose_coefs, motion_coefs, c=metric_vals, cmap='viridis_r')
    ax.set_xlabel('Pose Guidance Coef')
    ax.set_ylabel('Motion Guidance Coef')
    ax.set_title(f'{metric} Sweep Plot')

    norm = plt.Normalize(low_val, high_val)
    cax = fig.add_axes([0.94, 0.1, 0.02, 0.75])  # [left, bottom, width 5% of figure width, height 75% of figure height]
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'), cax=cax, orientation='vertical')

    #plt.subplots_adjust(right=0.15)
    plt.savefig(os.path.join(test_folder, f"{metric} Sweep Plot"), bbox_inches='tight')
    plt.close()
