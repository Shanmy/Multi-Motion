This sets up Mistral-7B-Instruct-v0.2-AWQ, which is 4-bit quantized version of Mistral-7B-Instruct-v0.2, using AutoAWQ quantization library
This code is tested on cs3, using "mitch/pose-data:v1" image (b476da8a4810). 
`docker run --gpus all -u $(id -u):$(id -g)  $(id -G | sed -e 's/\</--group-add /g')  -v "/home/us000217/":/tmp -it mitch/pose-data:v1`

#############################################
How to use: 
1. set up environment: install autoawq 0.1.6 using CUDA 11.8. Note default 'pip install autoawq' will install autoawq v0.1.8, using cuda 12, which will cause error
`pip install https://github.com/casper-hansen/AutoAWQ/releases/download/v0.1.6/autoawq-0.1.6+cu118-cp310-cp310-linux_x86_64.whl`

2. import `mistral_utils.py` and use simple init and run functions, as showed in `demo_run_mistral.py`

3. (Optional) If encountered the following error when run in docker, `export TRANSFORMERS_CACHE=/path/to/writable/folder` in bash, and run again should clear the error. 
`There was a problem when trying to write in your cache folder (/.cache/huggingface/hub). You should set the environment variable TRANSFORMERS_CACHE to a writable directory.
Traceback (most recent call last):
  File "/tmp/mistral_7b_instruct_v0.2_awq.py", line 5, in <module>
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
  File "/opt/conda/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 718, in from_pretrained
    tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 550, in get_tokenizer_config
    resolved_config_file = cached_file(
  File "/opt/conda/lib/python3.10/site-packages/transformers/utils/hub.py", line 430, in cached_file
    resolved_file = hf_hub_download(
  File "/opt/conda/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 1159, in hf_hub_download
    os.makedirs(storage_folder, exist_ok=True)
  File "/opt/conda/lib/python3.10/os.py", line 215, in makedirs
    makedirs(head, exist_ok=exist_ok)
  File "/opt/conda/lib/python3.10/os.py", line 215, in makedirs
    makedirs(head, exist_ok=exist_ok)
  File "/opt/conda/lib/python3.10/os.py", line 215, in makedirs
    makedirs(head, exist_ok=exist_ok)
  File "/opt/conda/lib/python3.10/os.py", line 225, in makedirs
    mkdir(name, mode)
PermissionError: [Errno 13] Permission denied: '/.cache'
`
