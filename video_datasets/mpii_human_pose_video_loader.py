import numpy as np
from torch.utils.data import Dataset

from video_datasets.video_data_utils import (
    prepare_video_frames,
)
from video_datasets.video_viz_util import viz_states
from pose_datasets.data_utils import list_files_recursively


# TODO: rewrite this in the manner of webvid code?

class MPIIHumanPoseVideoDataset(Dataset):
    def __init__(
        self,
        frames,
        image_size=None,
        world_rank=0,
        world_size=1,
        #data_dir='/nfs/USRCSEA/IVA/Datasets/mpii_human_pose_source_videos_ffmpeg',
        #data_dir='/nfs/USRCSEA/IVA/Datasets/mpii_human_pose_video_clips',
        #data_dir='/nfs/USRCSEA/IVA/Datasets/mpii_demo_data',
        data_dir='/nfs/USRCSEA/IVA/Datasets/mpii_human_pose_video_clips_v1',
        resize=False,
        random_crop=False,
        random_flip=False,
        frameskip=1,  # sample rate of video. frameskip 1 corresponds to original video. frameskip 2 will select every other frame, etc.
    ):
        super().__init__()
        if image_size is not None and type(image_size) is int:
            image_size = [image_size, image_size]
        self.resolution = image_size
        self.frames = (frames if frames is not None else 1)
        assert not (image_size is None and resize)

        video_paths = list_files_recursively(data_dir, file_types=None)
        self.local_videos = video_paths[world_rank:][::world_size]

        self.resize = resize
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.frameskip = frameskip

    def __len__(self):
        return len(self.local_videos)

    def __getitem__(self, idx):

        path = self.local_videos[idx]
        arr = prepare_video_frames(
                path=path,
                resolution=self.resolution,
                frames=self.frames,
                frameskip=self.frameskip,
                resize=self.resize,
                random_crop=self.random_crop,
                random_flip=self.random_flip,
            )

        out_arr = np.transpose(arr, [3, 0, 1, 2])
        return out_arr


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from video_viz_util import viz_states
    mpii_video_dataset = MPIIHumanPoseVideoDataset(frames=5)
    mpii_video_dataloader = DataLoader(
        mpii_video_dataset,
        batch_size=1,
        shuffle=True,
    )
    mpii_video_iterator = iter(mpii_video_dataloader)

    tries = 0
    successes = 0
    max_loads = 20
    for _ in range(len(mpii_video_dataloader)):
        if tries >= max_loads:
            break
        tries += 1
        try:
            mpii_video_sample = next(mpii_video_iterator)
            successes += 1
        except:
            continue
    
    print(f'{tries} loads attempted, {successes} loads succeeded.')
    viz_states('mpii_human_pose_clips/', mpii_video_sample)
    viz_states('mpii_human_pose_clips/', mpii_video_sample)
