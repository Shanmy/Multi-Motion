# python3 -m pip install -U yt-dlp

import os
from scipy import io


OUT_FOLDER = '/nfs/USRCSEA/IVA/Datasets/mpii_human_pose_source_videos_ffmpeg'

# data file from: http://human-pose.mpi-inf.mpg.de/#download
mat = io.loadmat('metadata/mpii_human_pose_v1_u12_1.mat')
vid_ids = mat['RELEASE'][0, 0][5][0]

for i in range(vid_ids.shape[0]):
    try:
        vid_id = vid_ids[i][0]
        youtube_str = f'https://www.youtube.com/watch?v={vid_id}'
        out_path = os.path.join(OUT_FOLDER, str(i).zfill(4))
        os.system(f'yt-dlp --external-downloader ffmpeg -o {out_path} {youtube_str}')
    except:
        pass
