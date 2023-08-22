## Setup

```sh
$ wget -P ~/ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod +x ~/Miniconda3-latest-Linux-x86_64.sh
$ ~/Miniconda3-latest-Linux-x86_64.sh -b
$ export PATH=~/miniconda3/bin:$PATH
$ conda init & conda config --set auto_activate_base false
# close and start a new session
$ conda activate base
$ conda install cudatoolkit=11.0 -y
```


```sh
# dependencies
$ pip install ipywidgets
$ pip install torch
$ pip install transformers
$ pip install sentencepiece
$ pip install datasets
# required to run 8bit training
$ pip install accelerate
$ pip install bitsandbytes
$ pip install peft
$ pip install scipy
# utils & server
$ pip install gradio
$ pip install pipx
# check GPU usage
$ pipx run nvitop
```


## Train

```sh
$ python train.py
$ WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 train.py
```