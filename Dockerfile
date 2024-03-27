FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update
RUN apt-get -y install tmux

COPY ./requirements.txt .
RUN pip install -r ./requirements.txt