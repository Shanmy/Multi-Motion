# python3 -m pip install -U yt-dlp

import os
from scipy import io


OUT_FOLDER = '/nfs/USRCSEA/IVA/Datasets/mpii_human_pose_video_clips'


def seconds_to_min_sec(seconds):
    mins = seconds // 60
    secs = seconds - (mins * 60)
    return mins, secs


def get_time_interval(vid_second_center, time_radius=10):
    vid_second_start = max(0, vid_second_center - time_radius)
    vid_second_end = max(vid_second_center + time_radius, 2 * time_radius)
    mins_start, secs_start = seconds_to_min_sec(vid_second_start)
    mins_end, secs_end = seconds_to_min_sec(vid_second_end)
    time_interval = f'{mins_start}:{str(secs_start).zfill(2)}-{mins_end}:{str(secs_end).zfill(2)}'
    return time_interval


# data file from: http://human-pose.mpi-inf.mpg.de/#download
mat = io.loadmat('metadata/mpii_human_pose_v1_u12_1.mat')
im_annotations = mat['RELEASE'][0, 0][0][0]
vid_ids = mat['RELEASE'][0, 0][5][0]

for i in range(im_annotations.shape[0]):
    print(f'Processing clip {i} of {im_annotations.shape[0]}')
    try:
        vid_id_num = im_annotations[i][3][0][0]  # subtract one bc matlab uses 1 as first array idx
        vid_id = vid_ids[vid_id_num - 1][0]
        vid_second_center = im_annotations[i][2][0][0]
    except:
        continue
    time_interval = get_time_interval(vid_second_center)
    print(vid_second_center)
    print(time_interval)
    youtube_str = f'https://www.youtube.com/watch?v={vid_id}'
    out_path = os.path.join(OUT_FOLDER, str(i).zfill(4))
    os.system(f'yt-dlp --force-keyframes-at-cuts --download-sections "*{time_interval}" -o {out_path} {youtube_str}')
