## LoRA


## Setup


```sh
# download the private key provided by Bes
$ chmod 600 ~/path/private/key
$ ssh -i ~/path/private/key <user>@<ip>
$ git clone https://github.com/besarthoxhaj/lora.git
```


```sh
$ wget -P ~/ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod +x ~/Miniconda3-latest-Linux-x86_64.sh
$ ~/Miniconda3-latest-Linux-x86_64.sh -b
$ export PATH=~/miniconda3/bin:$PATH
$ conda init
# close and start a new session
$ conda config --set auto_activate_base false
```


```sh
# conda environment
$ conda create --name lora python=3.8 -y
$ conda env list
$ conda activate lora
$ conda list
# dependencies
$ conda install cudatoolkit=11.0 -y
$ pip install ipywidgets
$ pip install fire
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
$ pipx run nvitop
```


## Train

```sh
$ python train.py
$ WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 train.py
```


## Dev

```sh
$ ls *.py | entr sh -c 'clear; python test_data.py'
```
