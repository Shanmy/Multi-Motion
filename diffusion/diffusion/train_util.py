import copy
import functools
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

import numpy as np
from .dist_util import dist_print, sync_params, read_text_prompts
from .fp16_util import MixedPrecisionTrainer
from .resample import LossAwareSampler, UniformSampler
from evaluations.fid import evaluate_from_trainer, get_encoder
from evaluations.save_eval_plots import save_all_eval_plots
from models.cfg_sampler import ClassifierFreeSampleModel
from viz.viz_utils import viz_from_loader

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        val_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        out_dir,
        local_rank,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        eval_num_samples=1024,
        eval_interval=None,
        eval_batch_size=None,
        get_eval_metrics=False,
        eval_max_viz_samples=None,
        eval_gt_pose_file=None,
        eval_gt_pose_text_file=None,
        eval_gt_motion_file=None,
        eval_gt_motion_text_file=None,
        model_1st_stage=None,
        eval_num_sample_frames=None,
        eval_pose_type='joint',
        guidance_param=None,
        mean_path="",
        std_path="",
        viz_frameskip=10,
    ):
        self.model = model
        self.model.train()
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.out_dir = out_dir
        self.log_file = os.path.join(self.out_dir, 'log', "log.txt")
        self.local_rank = local_rank
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.eval_num_samples = eval_num_samples
        self.eval_interval = eval_interval
        self.eval_batch_size = eval_batch_size
        self.get_eval_metrics = get_eval_metrics
        self.eval_max_viz_samples = eval_max_viz_samples
        self.eval_gt_pose_file = eval_gt_pose_file
        self.eval_gt_pose_text_file = eval_gt_pose_text_file
        self.eval_gt_motion_file = eval_gt_motion_file
        self.eval_gt_motion_text_file = eval_gt_motion_text_file
        self.model_1st_stage = model_1st_stage
        self.eval_num_sample_frames = eval_num_sample_frames
        self.eval_pose_type = eval_pose_type

        self.guidance_param = guidance_param
        self.mean_path = mean_path
        self.std_path = std_path

        self.step = 0
        self.resume_step = 0

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                dist_print(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        # =================== Load evaluation components ================
        self.viz_frameskip = viz_frameskip
        if self.eval_interval is not None:
            if self.get_eval_metrics:
                assert self.eval_gt_pose_file is not None or self.eval_gt_motion_file is not None
                if self.eval_gt_pose_file is not None:
                    self.gt_pose = torch.tensor(np.load(self.eval_gt_pose_file))
                    self.gt_pose_text = read_text_prompts(self.eval_gt_pose_text_file)
                    self.encoder_clpp = get_encoder('clpp', device=self.local_rank)
                if self.eval_gt_motion_file is not None:
                    self.gt_motion = torch.tensor(np.load(self.eval_gt_motion_file))
                    self.gt_motion_text = read_text_prompts(self.eval_gt_motion_text_file)
                    self.encoder_clmp = get_encoder('clmp', device=self.local_rank)
            if self.guidance_param is not None:
                self.cfg_model = ClassifierFreeSampleModel(self.model, self.guidance_param)
            else:
                self.cfg_model = None

    def load_model_wo_clip(self, model, state_dict): 
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False) 
        assert len(unexpected_keys) == 0 
        assert all([k.startswith('clip_model.') for k in missing_keys]) 

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                dist_print(f"loading model from checkpoint: {resume_checkpoint}", save_dir=self.log_file)
                state_dict = torch.load(resume_checkpoint, map_location=torch.device(self.local_rank))
                self.load_model_wo_clip(self.model, state_dict)
        sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                dist_print(f"loading EMA from checkpoint: {ema_checkpoint}", save_dir=self.log_file)
                state_dict = torch.load(ema_checkpoint, map_location=torch.device(self.local_rank))
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if os.path.exists(opt_checkpoint):
            if dist.get_rank() == 0:
                dist_print(f"loading optimizer state from checkpoint: {opt_checkpoint}", save_dir=self.log_file)
            state_dict = torch.load(opt_checkpoint, map_location=torch.device(self.local_rank))
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            time_start = time.time()

            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if (self.step + 1) % self.save_interval == 0:
                self.save()
            if self.eval_interval is not None and (self.step + 1) % self.eval_interval == 0:
                self.model.eval()
                if self.cfg_model is not None:
                    self.cfg_model.eval()
                self.run_eval()
                self.model.train()
                if self.cfg_model is not None:
                    self.cfg_model.train()

            time_per_step = time.time() - time_start
            if self.step == 0 or (self.step + 1) % self.log_interval == 0:
                dist_print(f'Time Per Step: {time_per_step:.3f}', save_dir=self.log_file)

            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if self.step % self.save_interval != 0:
            self.step -= 1
            self.save()
            self.step += 1

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(self.local_rank)
            micro_cond = {
                k: (
                    v[i : i + self.microbatch].to(self.local_rank) if torch.is_tensor(v)
                    else v[i : i + self.microbatch]
                )
                for k, v in cond.items()
            }
            if hasattr(self.model, 'update_cond_dict'):
                # training conditioned on data
                micro_cond = self.model.update_cond_dict(micro, micro_cond)
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.local_rank)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.mp_trainer.backward(loss)

        if self.step == 0 or (self.step + 1) % self.log_interval == 0:
            dist_print(f'Step {self.step+1}. Loss={loss.item():.7f}', save_dir=self.log_file)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def run_eval(self):
        # TODO: add metrics back in. for now, only do viz during eval.
        poses, cond, metric_report = evaluate_from_trainer(self)
        dist_print(metric_report, save_dir=self.log_file)
        if dist.get_rank() == 0:
            save_all_eval_plots(self.log_file, frequency=self.eval_interval)
            sample_folder = os.path.join(self.out_dir, 'viz', str(self.step+1).zfill(7))
            os.makedirs(sample_folder)
            # NOTE: already unnormalized the poses in "sample_diffusion" inside "evaluate_from_trainer"
            unsqueeze_pose_dim = (self.eval_pose_type == 'motion')
            unsqueeze_motion_dim = (self.eval_pose_type == 'multipose')
            eval_max_viz_samples = min(self.eval_max_viz_samples, self.eval_batch_size)
            viz_from_loader(
                poses=poses[0:eval_max_viz_samples],
                cond_dict={k: v[0:eval_max_viz_samples] for k, v in cond.items()},
                save_dir=sample_folder,
                unsqueeze_pose_dim=unsqueeze_pose_dim,
                unsqueeze_motion_dim=unsqueeze_motion_dim,
                frameskip=self.viz_frameskip,
                mean_path=self.mean_path,
                std_path=self.std_path,
            )

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                dist_print(f"saving model {rate}...", save_dir=self.log_file)
                if not rate:
                    filename = f"model{(self.step+self.resume_step+1):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step+1):06d}.pt"
                with open(os.path.join(self.out_dir, 'checkpoints', filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with open(
                os.path.join(self.out_dir, 'checkpoints', f"opt{(self.step+self.resume_step+1):06d}.pt"),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = os.path.join(os.path.dirname(main_checkpoint), filename)
    if os.path.exists(path):
        return path
    return None


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
