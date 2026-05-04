import numpy as np

from torch.utils.data import Dataset, DataLoader


class ValPromptDataset(Dataset):
    def __init__(
        self,
        lengths=61,
        #val_text="../pose_datasets/laion_val.txt",
        val_text="../pose_datasets/laion_viz_prompts.txt",
        mean_path=None,
        std_path=None,
    ):
        super().__init__()

        with open(val_text) as f:
            self.prompt_lists = f.readlines()
        self.lengths = lengths
        self.mean_path = mean_path
        self.std_path = std_path

    def __len__(self):
        return len(self.prompt_lists)

    def __getitem__(self, idx):
        prompt = self.prompt_lists[idx][:-3]
        try:
            num_poses = int(self.prompt_lists[idx].replace('\n', '')[-1])
        except:
            num_poses = 2
        dummy_pose = np.zeros((self.lengths, 10, 158))
        cond = {'text': prompt, 'num_poses': num_poses, 'lengths': self.lengths}
        
        return dummy_pose, cond


if __name__ == "__main__":
    val_prompt_dataset = ValPromptDataset()
    val_prompt_dataloader = DataLoader(val_prompt_dataset, batch_size=10, shuffle=False)
    for i, (pose, cond) in enumerate(val_prompt_dataloader):
        breakpoint()
