import os
import sys
try:
    import bpy
    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError("Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender.")

import temos_render.launch_blender
import temos_render.prepare  # noqa
import logging
import hydra
from omegaconf import DictConfig

from glob import glob

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="temos_render", config_name="render_config")
def _render_cli(cfg: DictConfig):
    return render_cli(cfg)


def extend_paths(path, keyids, *, onesample=True, number_of_samples=1):
    if not onesample:
        template_path = str(path / "KEYID_INDEX.npy")
        paths = [template_path.replace("INDEX", str(index)) for index in range(number_of_samples)]
    else:
        paths = [str(path / "KEYID.npy")]

    all_paths = []
    for path in paths:
        all_paths.extend([path.replace("KEYID", keyid) for keyid in keyids])
    return all_paths


def render_cli(cfg: DictConfig) -> None:
    if cfg.npy is None:
        # "/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mengyi-v3/joint_8_gpu/24-01-25-03-47-44/samples/joint_single_1_gpu/24-02-02-02-26-16/sample_00000"
        paths = glob(f"{cfg.folder}/*/vertices.npy")
    else:
        paths = [cfg.npy]

    from temos_render.blender import render
    from temos_render.video import Video
    import numpy as np

    init = True
    for path in paths:
        try:
            data = np.load(path)
        except FileNotFoundError:
            logger.info(f"{path} not found")
            continue

        frames_folder = path.replace(".npy", ".png")

        out = render(data, frames_folder,
                     denoising=cfg.denoising,
                     oldrender=cfg.oldrender,
                     res=cfg.res,
                     canonicalize=cfg.canonicalize,
                     exact_frame=cfg.exact_frame,
                     num=data.shape[0], mode=cfg.mode,
                     faces_path=cfg.faces_path,
                     downsample=cfg.downsample,
                     always_on_floor=cfg.always_on_floor,
                     init=init,
                     gt=cfg.gt)

        init = False
        logger.info(f"Frame generated at: {out}")


if __name__ == '__main__':
    _render_cli()