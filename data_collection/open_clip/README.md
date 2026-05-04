## Train CLIP Pose Model

This code base is the same as [OpenCLIP](https://github.com/mlfoundations/open_clip), except for the following files which are added to perform text/pose pretraining:

* ```src/training/main_pose.py```
* ```src/training/data_pose.py```
* ```src/training/model_pose.py```

Train the model using the command:

```
bash open_clip_pose.sh
```


## Train CLIP Image Model (validation of codebase)

```
bash test_open_clip_rn50.sh
```