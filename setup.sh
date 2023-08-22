wget -P ~/ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/Miniconda3-latest-Linux-x86_64.sh
~/Miniconda3-latest-Linux-x86_64.sh -b
export PATH=~/miniconda3/bin:$PATH
conda init & conda config --set auto_activate_base false

conda activate base
conda install cudatoolkit=11.0 -y
pip install torch sentencepiece datasets bitsandbytes peft accelerate scipy pipx