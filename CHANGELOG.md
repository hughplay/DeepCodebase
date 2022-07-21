# Changelog

>**TODO List and Changes of Code**
>
>When building projects, one of best practices is to list TODOs as a reminder for yourselves and record the major changes. It helps you to save lots of time from struggling to remember why some codes are organized like the way you see after a long time. It may also help others to understand your code, if you would like to share them to others.
>
> If you are interested in learning more about changelog, read this:
> https://keepachangelog.com/


## TODO

- [ ] spell check
- [ ] add some logos to the teaser, including PyTorch, Lightning, ...

## 2022-7-21 15:39:19

- [x] documents
    - [x] README.md
    - [x] CHANGELOG.md
    - [x] DEVELOPMENT.md

## 2022-7-20 13:50:30

- [x] add sphinx api docs

## 2022-7-19 21:43:29

- [x] precommit hooks is great! ;)
- [x] add setup.cfg for linter configurations

## 2022-7-17 17:06:04

- [x] test resume training
- [x] test `test.py`
- [ ] May explore hparams search in the future.

The name of logger all becomes rank_zero which is not good.
## 2022-7-17 16:43:24

Make `print_config` ordered by recreating a DictConfig with ordered keys.
`print_config` in this project is different with the one in hydra-lightning-template. The color of the one in hydra-lightning-template is not my favorite.


## 2022-7-17 15:21:52

- remove dataset.name.
- add `name`
- add `exp_id`: `${dataset._target_}_${model._target_}_${criterion._target_}_${now:%Y-%m-%d-%H-%M-%S}`
- add omegaconf resolver, tail
- `exp_id` becomes `${tail:dataset._target_}_${tail:model._target_}_${tail:criterion._target_}_${now:%Y-%m-%d-%H-%M-%S}`

**Question: loguru or logging? Do we need to use loguru?**

Seems that logging has no obvious weakness. And it may waste time to figure out how to use loguru with PytorchLignightning.

- [x] logging
- [x] save hparams to `config.yaml` and wandb beautifully. :)


## 2022-07-15 08:42:41

- [x] requirements.txt
- [x] confiuration
    - [x] train.yaml
    - [x] dataset
    - [x] model
    - [x] optim
    - [x] scheduler
    - [x] pipeline
    - [x] pl_trainer
    - [x] logging
    - [x] optim
    - [x] logdir
    - [x] callbacks
    - [x] experiment
    - [x] debug
    - [x] optional local
- [x] pipeline
- [x] train.py


## 2022-07-12 00:28:32

- [x] build new docker
    - simple docker file: `/docker/Dockerfile`
    - full docker file: `/docker/Dockerfile.full`
    - first three sections are built into [`deepbase/project:codebase`](https://hub.docker.com/r/deepbase/project/tags/)
    - simple docker file use `FROM deepbase/project:codebase`


## 2022-7-8 15:16:17

- [x] modify README.md
- [x] add conf/local according to hydra-lightning-template


## 2022-7-7 17:54:52

- [x] sample models: DNN, LeNet
- [x] add unit tests, refer to:
    - https://github.com/ashleve/lightning-hydra-template#tests
    - https://docs.pytest.org/en/7.1.x/getting-started.html
    - read [Packaging a python library](https://packaging.python.org/tutorials/packaging-projects/)
    - `pytest -s` to show print messages.
- [x] `test.py` ready.


## 2022-7-4 09:47:48

Start to prepare code folders by referring my previous projects and
lightning-hydra-template.

- [x] make MNIST datamodule ready


**Where to place the loss computing? Model or Lightning Module?**

In my previous experience, I placed the loss computing in Lightning Module.
However, it is very common to test different loss function in the same task.
While I think a `LightningModule` is corresponding to a specific task, such as
classification or generation, the main content of a `LightningModule` should
only decide training pipeline, e.g. training logic, evaluation metrics, logging,
etc. Due to this reason, I think it is better to place the loss computing in the
model to make `LightningModule` more general to fit different models. Actually,
loss function design is actually a part of the model design in many deep
learning papers.


## 2022-7-3

- [x] README.md, including template, icons, ...
