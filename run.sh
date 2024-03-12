#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/ser_%A.out
#SBATCH --job-name=ser
#SBATCH -n 1

# !mkdir -p logs # to create ./logs/

module load miniconda
module load cuda/11.8 
module load ffmpeg/4.3.2

source activate mmer 

# setup
conda create -n mmer 
conda install pip
pip3 install --upgrade pip
pip install git+https://github.com/huggingface/datasets.git
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt

## Speech Emotion Recognition
# - training ser_seq seems to take long 10hr and eval_acc. not improving, epoch seems off.
# - OpenFace needs to be compiled for unix

## Facial Emotion Recognition

## Fusion

## Biosignal Inversion

