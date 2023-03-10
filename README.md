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


TLDR; We have prepare the commands in the `Makefile` for you:

```bash
# use the template and clone the new project to local
# change the directory to the new project
make init           # run in host
make in             # run in host
make wandb_login
make train
make table
# ctrl + d to exit the container
make copy_git      # run in host
make in            # run in host
# modify the project and prepare to commit
make precommit_install
make commit
```

Below is the detailed instructions.

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
python docker.py startd --build
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

or launch multiple training at once:

```bash
python scripts/batch_train.py scripts/training/train_baselines.sh
```

**6. View results on wandb**

The training results will be automatically uploaded to wandb. You can view the results on [wandb.ai](https://wandb.ai).

**7. Generate Latex table**

After training multiple models, you can generate a latex table with:

```bash
python scripts/gen_latex_table.py --table baseline
```

This will generate a table like this:

```latex
\usepackage{booktabs}

\begin{table}[t]
\centering
\small
\caption{Performance on the MNIST dataset. }
\label{tab:baseline}
\begin{tabular}{lr}
\toprule
Model & Accuracy \\
\midrule
DNN & 0.974566 \\
LeNet & \textbf{0.983000} \\
\bottomrule
\end{tabular}
\end{table}
```

**8. Custom your own project**

Modify the code to fit your own deep learning project. Check our [document](https://hongxin19.github.io/deepcodebase/) for more details.


**9. Commit you modifications**

We strongly recommend you to use [pre-commit](https://pre-commit.com/) to check your code before commiting. Before you first commit, you need to install the pre-commit hook with:

```bash
pre-commit install
```

Then, you can commit your modifications with normal git commands.

```bash
git add .
git commit -m "commit message"
```

It will show you the errors if your code does not pass the pre-commit checks. After you fix the errors, you need to git add the modified files and commit again.


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

## Projects based on DeepCodebase

- [TVR](https://github.com/hughplay/TVR)

## License

DeepCodebase is released under the [MIT license](LICENSE).
