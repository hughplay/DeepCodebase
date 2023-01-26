<div align = center>

<br>

<img src = "docs/_static/imgs/logo.svg" width="80%">

<br>
<br>

<!-- [![DeepCodebase](docs/_static/imgs/logo.svg)](https://github.com/hughplay/DeepCodebase) -->

<!-- https://ileriayo.github.io/markdown-badges/ -->
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square&logo=docker&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=flat-square&logo=anaconda&logoColor=white)
[![](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![](https://img.shields.io/badge/-Lightning-792ee5?style=flat-square&logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![](docs/_static/imgs/hydra.svg)](https://hydra.cc)
![](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=flat-square&logo=WeightsAndBiases&logoColor=black)
[![](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square&labelColor=gray)](#license)

*A template for deep learning projects.*

[[Use this template]](https://github.com/hughplay/DeepCodebase/generate) [[Document]](https://hongxin2019.github.io/deepcodebase/)

</div>

<br>

## Pre-requirements

- [Docker](https://docs.docker.com/engine/install)
- [Nvidia-Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- [docker-compose](https://docs.docker.com/compose/install/)

<!-- <br> -->


## Quick Start

**1. Create new project from template**

Click [[Use this template]](https://github.com/hughplay/DeepCodebase/generate) to create a new project from this template.

**2. Clone the new project to local**

You can find the repository url in the new project page.

```bash
git clone <repo url>
```

**3. Prepare docker environment**


Build the docker image and launch the container with:

```bash
python docker.py startd
```

**4. Enter the docker container**

Enter the container with:

```bash
python docker.py
```

The docker container is our development environment.

**5. Train a model**

Launch training with:

```bash
python train.py experiment=mnist_dnn
```

**6. View results on wandb**

The training results will be automatically uploaded to wandb. You can view the results on [wandb.ai](https://wandb.ai).

**7. Custom your own project**

...

Check our [document](https://hongxin19.github.io/deepcodebase/) for more details.


## Cite us

Give a star and cite us if you find this project useful.

[![DeepCodebase](https://img.shields.io/badge/Deep-Codebase-2d50a5.svg?style=flat-square)](https://github.com/hughplay/DeepCodebase)

```md
[![DeepCodebase](https://img.shields.io/badge/Deep-Codebase-2d50a5.svg?style=flat-square)](https://github.com/hughplay/DeepCodebase)
```

*This is a project based on [DeepCodebase](https://github.com/hughplay/DeepCodebase) template.*

``` md
*This is a project based on [DeepCodebase](https://github.com/hughplay/DeepCodebase) template.*
```

## License

DeepCodebase is released under the [MIT license](LICENSE).
