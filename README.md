# DeepCodebase

Official repository for ["DeepCodebase for Deep Learning"](https://github.com/hughplay/DeepCodebase).

![A fancy image here](docs/_static/imgs/logo.svg)

**Figure:** *DeepCodebase for Deep Learning. (Provide a fancy image here to impress your audience.)*

> **DeepCodebase for Deep Learning** <br>
> Xin Hong <br>
> *Published on Github*

[![](docs/_static/imgs/project.svg)](https://hongxin2019.github.io)
[![](https://img.shields.io/badge/-code-green?style=flat-square&logo=github&labelColor=gray)](https://github.com/hughplay/DeepCodebase)
[![](https://img.shields.io/badge/arXiv-1234.5678-b31b1b?style=flat-square)](https://arxiv.org)
[![](https://img.shields.io/badge/Open_in_Colab-blue?style=flat-square&logo=google-colab&labelColor=gray)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
[![](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![](https://img.shields.io/badge/-Lightning-792ee5?style=flat-square&logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![](docs/_static/imgs/hydra.svg)](https://hydra.cc)

## News

- [x] **[2022-07-21]** Initial release of the DeepCodebase.

## Description

DeepCodebase is a codebase/template for deep learning researchers, so that do
experiments and releasing projects becomes easier.
**Do right things with suitable tools!**

This README.md is meant to be the template README of the releasing project.
**[Read the Development Guide](./DEVELOPMENT.md) to realize and start to use DeepCodebase**

If you find this code useful, please star this repo and cite us:

```
@inproceedings{deepcodebase,
  title={DeepCodebase for Deep Learning},
  author={Xin Hong},
  booktitle={Github},
  year={2022}
}
```

## Environment Setup

This project recommends using [docker](https://www.docker.com/) to run experiments.

### Quick Start

The following steps are to build a docker image for this project and run it.

**Step 1.** Install docker-compose in your workspace.
```sh
# (set PyPI mirror is optional)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install docker-compose
```

**Step 2.** Build a docker image and start a docker container.
```sh
python docker.py prepare --build
```

**Step 3.** Enter the docker container at any time, start experiments now.
```sh
python docker.py [enter]
```

### Data Preparation

MNIST will be automatically downloaded to `DATA_ROOT` and prepared by `torch.dataset.MNIST`.


## Training

Commonly used training commands:

```sh
# training a mnist_lenet on GPU 0
python train.py experiment=mnist_lenet devices="[0]"

# training a mnist_lenet on GPU 1
python train.py experiment=mnist_dnn devices="[1]"

# training a mnist_lenet on two gpus
python train.py experiment=mnist_lenet devices="[2,3]" name="mnist lenet 2gpus"
```

## Testing

Commonly used testing commands:

```sh
# * test the model, <logdir> has been printed twice (start & end) when training
python test.py <logdir>
# test the model, with multiple config overrides, e.g.: to test multiple datasets
python test.py <logdir> --update_func test_original test_example
# update wandb, and prefix the metrics
python test.py --update_func test_original test_example --prefix original example --update_wandb
# generate LaTex Tables
python scripts/generate_latex_table.py
```

## Acknowledgement

Many best practices are learned from [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), thanks to the maintainers of this project.

## License

[MIT License](./LICENSE)
