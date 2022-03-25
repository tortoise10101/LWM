FROM nvidia/cuda:11.0-devel-ubuntu20.04

RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip
RUN apt install -y vim

WORKDIR /home/
COPY . .
RUN pip install -r requirements.txt
