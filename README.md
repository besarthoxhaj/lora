## LoRA


## Setup


```sh
$ chmod 700 ~/.ssh/id_ed25519
$ chmod 700 ~/.ssh/id_ed25519.pub
$ git clone git@github.com:besarthoxhaj/lora.git
$ git clone https://github.com/tloen/alpaca-lora.git
$ git config --global user.email "besartshyti@gmail.com"
$ git config --global user.name "bes - fluidstack"
```


```sh
$ wget -P ~/ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod +x ~/Miniconda3-latest-Linux-x86_64.sh
$ ~/Miniconda3-latest-Linux-x86_64.sh -b
# accept the Terms of Service and start a new session
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
# utils
$ pip install pipx
$ pipx run nvitop
```


## Train

```sh
$ WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 finetune.py --base_model 'decapoda-research/llama-7b-hf' --data_path 'yahma/alpaca-cleaned' --output_dir './lora-alpaca'
```


## Dev

```sh
$ ls *.py | entr sh -c 'clear; python test_data.py'
```
