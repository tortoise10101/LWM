#FROM nvidia/cuda:11.0-devel-ubuntu20.04
FROM nablascom/cuda-pytorch:latest

WORKDIR /home/
COPY . .
