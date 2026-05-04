# Mulitperson Motion Diffusion (Updated 12/17/23)

Edit of [guided-diffusion](https://github.com/openai/guided-diffusion) repo to use standard DDP. This can serve as the foundation of training video diffusion models or other kinds of diffusion models. Support for both single-node and multi-node DDP.

## Training Job

To run a training job, need to specify the number of GPUs, config file, and number of nodes (if doing multinode training).

Example run command to train using ```configs_train/joint.py``` using 4 GPU on a single node:
```
bash train_one_node.sh 4 configs_train/joint.py
```

Example run command to train using ```configs_train/joint.py``` using 32 GPUs with 4 nodes and 8 GPUs per node:
```
bash train_multi_node.sh 8 4 configs_train/joint.py
```

## Test-Time Sampling and Eval

Executable files for sampling are similar to those for training. Example of sampling from trained model using 8 GPUs and ```configs_sample/joint.py```:

```
bash sample_one_node.sh 8 configs_sample/joint.py
```

TODO: Update ```get_eval_metrics.py``` to work with FID for motion/pose.

## Configs

Each file will have a corresponding training config in ```configs_train``` and sampling config in ```configs_sample```.

* ```joint.py```: Train multi-person motion model using AMASS and LAION Pose data.
* ```motion.py```: Train single person motion model using AMASS data.
* ```multi_pose.py```: Train multi-person pose model using LAION Pose data.
