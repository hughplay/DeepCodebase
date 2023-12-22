# ref: https://saattrupdan.github.io/2022-08-28-makefu/
# ref: https://gist.github.com/MarkWarneke/2e26d7caef237042e9374ebf564517ad
-include .env

# run on host
init:
	python docker.py startd --build

# run on host
in:
	python docker.py

wandb_login:
	wandb login

train:
	python train.py experiment=mnist_dnn

table:
	python scripts/gen_latex_table.py --table baseline

# run on host
copy_git:
	cp ~/.gitconfig $(CONTAINER_HOME)/ && cp -r ~/.ssh $(CONTAINER_HOME)/

precommit_install:
	pre-commit install

commit:
	git add .
	git commit

push:
	git push

nsightrun:
	nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true --cudabacktrace='all:10000' --osrt-threshold=10000 -x true $(cmd)

jupyter:
	jupyter lab --ip 0.0.0.0
