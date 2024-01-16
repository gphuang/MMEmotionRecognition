#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/fer_static_%A.out
#SBATCH --job-name=fer
#SBATCH -n 1

# !kinit # to access data on /teamwork/
# !mkdir -p logs # to create ./logs/

module load miniconda
module load ffmpeg/4.3.2
module load gcc/8.4.0 
module laod cuda

source activate mmer # created using requirement file: torchaudio version 0.10

## Facial Emotion Recognition

### Pre-processing
# To extract the Action Units (AUs) using the OpenFace library, we run: 
python3 ./src/Video/OpenFace/AUsFeatureExtractor.py \
            --videos_dir /teamwork/t40511/emotion_av/RAVDESS/video_speech \
            --openFace_path /scratch/work/huangg5/tutorials/OpenFace \
            --out_dir ./data/Extracted_AUs \
            --out_dir_processed ./data/processed_AUs

### Training Static Models

python3 ./src/Video/models/staticModels/FeatureTrainingAUs.py \
    --AUs_dir ./data/processed_AUs \
    --model_number 11 \
    --param (80) \
    --type_of_norm 1 \
    --out_dir ./data/models/avg_MLP80_AUs/posteriors

