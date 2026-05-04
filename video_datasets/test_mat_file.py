# python3 -m pip install -U yt-dlp

import os
import random
from scipy import io

# data file from: http://human-pose.mpi-inf.mpg.de/#download
mat = io.loadmat('metadata/mpii_human_pose_v1_u12_1.mat')
im_annotations = mat['RELEASE'][0, 0][0][0]
vid_ids = mat['RELEASE'][0, 0][5][0]
cats = mat['RELEASE'][0, 0][4]

def seconds_to_min_sec(seconds):
    mins = seconds // 60
    secs = seconds - (mins * 60)
    return mins, secs

# example
found_sample = False
while not found_sample:
    sample_id = random.randint(0, im_annotations.shape[0])
    print('Getting video for sample: ', sample_id)
    try:
        vid_id_num = im_annotations[sample_id][3][0][0]  # subtract one bc matlab uses 1 as first array idx
        vid_id = vid_ids[vid_id_num - 1][0]
        vid_second_center = im_annotations[sample_id][2][0][0]
        found_sample = True
    except:
        pass

youtube_str = f'https://www.youtube.com/watch?v={vid_id}'
print(youtube_str)
print(cats[sample_id])

time_radius = 5  # fwd and backward search in seconds
vid_second_start = max(0, vid_second_center - time_radius)
vid_second_end = max(vid_second_center + time_radius, 2 * time_radius)
mins_start, secs_start = seconds_to_min_sec(vid_second_start)
mins_end, secs_end = seconds_to_min_sec(vid_second_end)
time_interval = f'{mins_start}:{str(secs_start).zfill(2)}-{mins_end}:{str(secs_end).zfill(2)}'
print(f'Clip Time Interval: {time_interval}')

out_name = 'mpii_human_pose_clips/mpii_test'
os.system(f'yt-dlp --force-keyframes-at-cuts --download-sections "*{time_interval}" -o {out_name} {youtube_str}')
#os.system(f'yt-dlp --downloader ffmpeg -o {out_name} {youtube_str}')
