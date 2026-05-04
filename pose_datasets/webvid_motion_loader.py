import numpy as np
import os
from glob import glob
import os
import random
from datetime import datetime
from tqdm import tqdm

import numpy as np
from scipy.ndimage import gaussian_filter1d

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from viz.viz_utils import viz_from_loader
from data_utils import (
    align_laion_pose,
    convert_smpl_to_6d,
    process_translations,
    make_shorter_instruct_blip_text,
    pose_random_rotation,
)


def process_one_video(output):
    ids = output['outputs'].item()['track_ids']
    frames = output['outputs'].item()['reorganize_idx']
    smpl_thetas = convert_smpl_to_6d(torch.from_numpy(output['outputs'].item()['smpl_thetas']))
    smpl_betas = np.concatenate((np.zeros((frames.shape[0], 1)), output['outputs'].item()['smpl_betas']), axis=1)
    trans = output['outputs'].item()['cam_trans']
    smpl = np.concatenate((smpl_betas, smpl_thetas, trans), axis=1)  # all smpls x 158
    smpl[:, 11:] = align_laion_pose(smpl[:, 11:])
    max_f = max(frames)
    max_n = max(ids)
    motion = np.zeros((max_f+1, max_n, 158))
    mask = np.zeros((max_f+1, max_n, 158))
    
    for i in range(frames.shape[0]):
        motion[frames[i], ids[i]-1] = smpl[i]
        mask[frames[i], ids[i]-1] = 1
        
    mask = (mask.sum(axis=2) / 158).astype(int)
    
    return motion, mask


def merge_ids(motion, mask, threshold=0.3):
    mask = mask.T
    motion = motion.transpose(1, 0, 2)
    removed_ids = []
    for i1 in range(motion.shape[0]):
        for i2 in range(i1 + 1, motion.shape[0]):
            if i1 not in removed_ids and i2 not in removed_ids and mask[i2].sum() > 0 and mask[i1].sum() > 0:
                last = np.where(mask[i1] == 1)[0][-1]
                first = np.where(mask[i2] == 1)[0][0]
                if np.power(motion[i1, last, -3:] - motion[i2, first, -3:], 2).sum() < threshold:
                    removed_ids.append(i2)
                    motion[i1, first:] = motion[i2, first:]
                    mask[i1, first:] = mask[i2, first:]
    motion = np.delete(motion, removed_ids, axis=0)
    mask = np.delete(mask, removed_ids, axis=0)
    return motion, mask


def merge_rows(mask, rows, threshold=30):
    """
    mask: r x c
    rows: list of row indices from 0 to r-1
    """
    shared = mask[rows[0]]
    for row in rows:
        shared = shared * mask[row]
    if shared.sum() > threshold:
        same_idx = np.where(shared == 1)[0]
        if same_idx[-1] - same_idx[0] == same_idx.shape[0] - 1:
            return same_idx[0], same_idx[-1]
    return 0, 0


def find_motion_clip(mask, min_frames=30):
    intersect = {}
    mask = mask.T
    
    for i1 in range(mask.shape[0]):
        for i2 in range(i1 + 1, mask.shape[0]):
            #print(i1, i2)
            start, end = merge_rows(mask, [i1, i2], threshold=min_frames)
            if start != end:
                intersect[(i1, i2)] = (start, end)
                
    # find three people sequences
    cont_merge = True

    while cont_merge:
        removed = []
        added = []
        cont_merge = False
        all_keys = list(intersect.keys())
        if len(all_keys) > 100:
            intersect = dict(sorted(intersect.items(), key=lambda item: item[1][1] - item[1][0])[-100:])
            all_keys = list(intersect.keys())
            
        for k1 in range(len(all_keys)):
            for k2 in range(k1 + 1, len(all_keys)):
                if set(all_keys[k1]) in removed or set(all_keys[k2]) in removed or \
                    len(set(all_keys[k1]).intersection(set(all_keys[k2]))) == 0:
                    continue
                rows = set(all_keys[k1]).union(set(all_keys[k2]))
                start, end = merge_rows(mask, list(rows))
                if end - start > min_frames:
                    removed.append(tuple(all_keys[k1]))
                    removed.append(tuple(all_keys[k2]))
                    added.append((tuple(rows), (start, end)))
                    cont_merge = True
            
        for rows, interval in set(added):
            intersect[rows] = interval
        delete = [intersect.pop(x) for x in set(removed)]
    return intersect


def process_all_videos(text_file, visualize=False, min_frames=30):
    with open(text_file, "r") as f:
        all_videos = f.readlines()
    
    for video in tqdm(all_videos[:]):

        video = video[:-1]
        video_id = video.split("/")[-1].split(".")[0]
        ext = video.split("/")[-1].split(".")[1]
        if os.path.isfile(f"{WEBVID_MOTION_PATH}/{video_id}/{video_id}.{ext}.npz"):
            output = np.load(f"{WEBVID_MOTION_PATH}/{video_id}/{video_id}.{ext}.npz", allow_pickle=True)
            motion, mask = process_one_video(output)

            ids = mask.sum(axis=0) > min_frames
            mask = mask[:, ids]
            motion = motion[:, ids]
            
            intersect = find_motion_clip(mask, min_frames=min_frames)
            intersect = list(intersect.items())

            for i, (row, (start, end)) in enumerate(intersect):
    
                cond_dict = {'lengths': np.array([end - start] * len(row)), 'text': [''], \
                    'num_poses': np.array([len(row)] * (end - start))}
                motion_clip = motion[start:end, np.array(list(row))][None, :, :]
                
                if visualize:
                
                    viz_from_loader(
                        poses=torch.Tensor(motion_clip),
                        cond_dict=cond_dict,
                        save_dir=save_dir,
                        frameskip=10,
                        mean_path=None,
                        std_path=None,
                        unsqueeze_pose_dim=False,
                        unsqueeze_motion_dim=False,
                    )
                
                os.makedirs(f"{WEBVID_MOTION_PATH}/{video_id}/motion/", exist_ok=True)
                np.save(f"{WEBVID_MOTION_PATH}/{video_id}/motion/{i:03d}", motion_clip[0])
                with open(f"{WEBVID_MOTION_PATH}/{video_id}/annot.txt", "a") as f:
                    row = [str(x) for x in row]
                    f.writelines([f"{i}, {' '.join(row)}, {start}, {end}\n"])

        
def count_motions():

    all_videos = os.listdir(WEBVID_MOTION_PATH)
    with open(f"{WEBVID_MOTION_PATH}/webvid_motion.txt", "w") as f:
        for video in all_videos:
            for motion in glob(f"{WEBVID_MOTION_PATH}/{video}/motion/*.npy"):
                f.writelines([motion + '\n'])


class WebVidMotionDataset(Dataset):
    def __init__(
        self,
        max_len=120,
        max_pose=10,
        motion_list_path="/nfs/USRCSEA/IVA/Datasets/trace_motion/webvid_filter_03.txt",  # filtering of raw data
        #motion_list_path="/nfs/USRCSEA/IVA/Datasets/trace_motion/webvid_motion.txt",  # original raw and messy webvid motion data
        use_dummy_betas=True,
        use_dummy_hand_joints=True,
        use_dummy_text=False,
        shorten_instruct_blip_text=True,
        normalize=False,
        mean_path=None,
        std_path=None,
        center_first_frame=True,
        randomly_rotate_pose=True,
        trans_aug_rad=None,
        smooth_scale=1,
    ):

        self.motion_list_path = motion_list_path
        with open(self.motion_list_path, "r") as f: 
            self.file_list = f.readlines()
        self.file_list.sort()
        self.normalize = normalize
        self.max_len = max_len
        self.max_pose = max_pose
        self.use_dummy_betas = use_dummy_betas
        self.use_dummy_hand_joints = use_dummy_hand_joints
        self.use_dummy_text = use_dummy_text
        self.shorten_instruct_blip_text = shorten_instruct_blip_text
        self.smooth_scale = smooth_scale

        self.trans_aug_rad = trans_aug_rad
        # set to True to center the translation of first frame. False will center across all frames.
        self.center_first_frame = center_first_frame
        self.randomly_rotate_pose = randomly_rotate_pose

        if self.normalize:
            try:
                self.mean = np.load(mean_path).reshape(1, 1, 158).astype(np.float32)
            except:
                print('Warning: Unable to find mean file. Using zero mean instead.')
                self.mean = np.zeros(shape=[1, 1, 158]).astype(np.float32)
            try:
                self.std = np.load(std_path).reshape(1, 1, 158).astype(np.float32)
            except:
                print('Warning: Unable to find st dev file. Using one std dev instead.')
                self.std = np.ones(shape=[1, 1, 158]).astype(np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        # Load pose, [f_num x 158]
        pose = np.load(self.file_list[idx].replace("\n", ""), allow_pickle=True)
        pose = pose.astype(np.float32)

        if self.use_dummy_betas:
            # set the beta elements to 0
            pose[:, :, :11] = 0
        if self.use_dummy_hand_joints:
            # set hand joint pose id matrix (since these were not collected by BEV)
            pose[:, :, -15:-3] = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])

        # smooth pose predictions over time
        pose = gaussian_filter1d(pose, sigma=self.smooth_scale, axis=0)

        # center and augment translations
        pose[:, :, -3:] = process_translations(
                                pose[:, :, -3:],
                                center_first_frame=self.center_first_frame,
                                trans_aug_rad=self.trans_aug_rad
                            )

        if self.randomly_rotate_pose:
            pose = pose_random_rotation(pose)

        # divide by mean and std
        if self.normalize:
            pose = (pose - self.mean) / self.std

        # add dummy poses to end of pose sequence so all sequences have same length
        pad_pose = np.zeros((self.max_len, self.max_pose, pose.shape[2])).astype(np.float32)
        pad_pose[:pose.shape[0], :pose.shape[1]] = pose[:min(pose.shape[0], self.max_len), :min(pose.shape[1], self.max_pose)]

        # Load text, randomized
        # print(self.file_list[idx])
        text_path = self.file_list[idx][:-15] + 'instruct_blip.txt'
        if self.use_dummy_text or not os.path.exists(text_path):
            caption = ""
        else:
            with open(text_path) as f:  # FIXME
                captions = []
                for line in f.readlines():
                    line_split = line.strip().split('#')[0]
                    captions.append(line_split)
                caption = random.choice(captions)
                caption = caption.lower()
                if self.shorten_instruct_blip_text:
                    caption = make_shorter_instruct_blip_text(caption)

        return pad_pose, {
                            'text': caption,
                            'lengths': np.array(min(self.max_len, pose.shape[0])),
                            'num_poses': np.array(min(self.max_pose, pose.shape[1])),
                        }


if __name__ == "__main__":

    # process_all_videos("/home/s9053168/code/pose_caption_dataset/webvid-unique.txt")
    # count_motions()

    save_dir = 'out_webvid_viz'
    save_dir = os.path.join(save_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))

    batch_size = 10

    webvid_dataset = WebVidMotionDataset(
                            normalize=False,
                            randomly_rotate_pose=True,
                        )
    webvid_dataloader = DataLoader(
                            webvid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1,
                        )
    webvid_iterator = iter(webvid_dataloader)

    pose, cond = next(webvid_iterator)

    viz_from_loader(
            poses=pose,
            cond_dict=cond,
            save_dir=save_dir,
            frameskip=10,
            mean_path=None,
            std_path=None,
            unsqueeze_pose_dim=False,
            unsqueeze_motion_dim=False,
        )
