#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

ENV http_proxy=http://172.24.209.222:2222
ENV https_proxy=http://172.24.209.222:2222
ENV no_proxy=localhost,127.0.0.1

# user and group args
ARG USERNAME='us000240'
# get the ids below by running command "echo $(id -u)" and "echo $(id -G)"
ARG UID=1000240
ARG GID=10000
ARG USRCSEAGID=11004
ARG DEBIAN_FRONTEND=noninteractive

RUN groupadd -g $GID g_base
RUN groupadd -g $USRCSEAGID g_usrcsea
RUN useradd -rm -d /home/$USERNAME -s /bin/bash -u $UID -g $GID -g $USRCSEAGID -G root,sudo $USERNAME

# command line tools
RUN apt-get update
RUN apt-get install -y ffmpeg tmux net-tools git

# python-related commands
RUN pip install matplotlib cython lap setuptools==59.5.0 opencv-python-headless cloudpickle numpy==1.25.2 pandas scikit-learn einops
RUN pip install ftfy regex accelerate huggingface_hub transformers tokenizers==0.14.0 fvcore omegaconf pycocotools werkzeug webdataset
RUN pip install trimesh smplx scipy
RUN pip install imageio[ffmpeg]
RUN pip install 'git+https://github.com/openai/CLIP.git'
RUN pip install 'git+https://github.com/otaheri/chamfer_distance'
#RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'  # install detectron2 via command line in docker env
#RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"  # might not work during docker build, run inside container instead?

# run this for global pip install inside container (otherwise will install in user dir and cause leakage)
RUN chmod -R 777 /opt

# workdir and port
WORKDIR /home/$USERNAME
EXPOSE 22
