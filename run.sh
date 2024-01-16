#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/ser_%A.out
#SBATCH --job-name=ser
#SBATCH -n 1

# !kinit # to access data on /teamwork/
# !mkdir -p logs # to create ./logs/

module load miniconda
module load ffmpeg/4.3.2
module load gcc/8.4.0 
module laod cuda

source activate mmer # created using requirement file: torchaudio version 0.10

## Speech Emotion Recognition
# - training ser_seq seems to take long 10hr and eval_acc. not improving, epoch seems off.
# - OpenFace needs to be compiled for unix

## Facial Emotion Recognition

## Fusion

## Biosignal Inversion

