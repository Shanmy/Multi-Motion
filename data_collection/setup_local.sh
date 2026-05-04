# proxy (local only)
# export http_proxy=http://172.24.206.4:3128
# export https_proxy=http://172.24.206.4:3128

# updated proxy
export http_proxy=http://172.24.209.222:2222
export https_proxy=http://172.24.209.222:2222

# romp/bev
cd ROMP/simple_romp
python3 setup.py install
