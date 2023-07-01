## LoRA

```sh
# Set up a virtual environment
$ sudo apt update
$ sudo apt install python3.8-venv
$ python3 -m venv env
# Install dependencies
$ pip install wheel
$ pip install ipywidgets
$ pip install torch --index-url https://download.pytorch.org/whl/cu118
$ pip install transformers
$ pip install sentencepiece
# Those are required to run 8bit training
$ pip install accelerate
$ pip install bitsandbytes
$ pip install peft
$ pip install scipy
```

```sh
# Check GPU usage
$ sudo apt update
$ sudo apt install python3-pip
$ python3 -m pip install --user pipx
$ python3 -m pipx run nvitop
```

## Dev

```sh
$ ls *.py | entr sh -c 'clear; python test_data.py'
```