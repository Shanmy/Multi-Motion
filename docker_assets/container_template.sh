#!/bin/bash

# default args for gpus and name
gpus=${1:-"0,1,2,3"}
name=${2:-"ebm-exp"}
image=${3:-"mitch/hdebm:v1"}
wd=${4:-"/home/us000240/"}

# name of container and devices in use
devices=\"device=$gpus\"
container_name=$name-gpus-"${gpus//[,]/-}"

# container setup
docker run -it \
	-e http_proxy=http://172.24.209.222:2222 \
	-e https_proxy=http://172.24.209.222:2222 \
	-e HOME=$HOME \
	-e USER=$USER \
	-u $(id -u):$(id -g) \
	--group-add 10000 \
	--group-add 11004 \
 	-v /nfs/home/:/home/ \
	-v /nfs/USRCSEA:/nfs/USRCSEA \
    -v /nfs/SHARE/dataset:/nfs/SHARE/dataset \
	-w $wd \
	--ipc=host \
	--gpus "$devices" \
	--name $container_name \
	--entrypoint bash \
	$image
