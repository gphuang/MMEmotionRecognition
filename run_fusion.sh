#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/fusion_%A.out
#SBATCH --job-name=fusion
#SBATCH -n 1

###Training & Evaluation

    python3 MMEmotionRecognition/src/Fusion/FusionTraining.py  \
    --embs_dir_wav2vec <RAVDESS_dir>/FineTuningWav2Vec2_posteriors/20211020_094500 \
    --embs_dir_biLSTM <RAVDESS_dir>/FUSION/wav2Vec_AUs/BiLSTM_AUS/posteriors \
    --embs_dir_MLP MMEmotionRecognition/data/posteriors/avg_MLP80_AUs/posteriors \
    --out_dir <RAVDESS_dir>/FUSION/posteriors \
    --model_number 2 \
    --param 1.0 \
    --type_of_norm 1