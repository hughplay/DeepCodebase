# ref: https://saattrupdan.github.io/2022-08-28-makefu/
# ref: https://gist.github.com/MarkWarneke/2e26d7caef237042e9374ebf564517ad
-include .env

# run on host, build and start container
init:
	python docker.py startd --build

# run on host, start container but do not build
start:
	python docker.py startd

# run on host, enter container
in:
	python docker.py

# run on host, enter container as root
root:
	python docker.py --root

# run on host
copy_git:
	cp ~/.gitconfig $(CONTAINER_HOME)/ && cp -r ~/.ssh $(CONTAINER_HOME)/
