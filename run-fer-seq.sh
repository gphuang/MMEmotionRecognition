#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/fer_seq_%A.out
#SBATCH --job-name=fer
#SBATCH -n 1

module load miniconda
module load ffmpeg 
module load gcc  
module load cuda

source activate mmer # created using requirement file: torchaudio version 0.10

## Facial Emotion Recognition

## 1. Create the configuration file in sequenceLearning/conf/RAVDESS_AUs.json. 

## 2. Create the datasets
python3 ./src/Video/models/sequenceLearning/frontend/RAVDESS_AUs/frontend_ravdess_5CV.py \
    --AUs_dir ./data/processed_AUs \
    --out_dir ./src/Video/models/sequenceLearning/datasets/RAVDESS_AUs

## 3. Run set-up Train the sequential model with the parameters specified in RAVDESS_AUs.json. (5CV)

python3 ./src/Video/models/sequenceLearning/workflow/run.py train \
    ./src/Video/models/sequenceLearning/conf/RAVDESS_AUs.json \
    --kfolds 5

## 4. Extract posteriors

python3 workflow/run.py inference \
    ./src/Video/models/sequenceLearning/conf/RAVDESS_AUs_posteriors.json \
    --kfolds 5 \
    --pretrained ./src/Video/models/sequenceLearning/out/trained

# mofify the RAVDESS_AUs_posteriors.json to extract the embeddings from the training and from the validation
# by changing the 'inference_data' parameter (See example below)
### Example RAVDESS_AUs_posteriors.json:
    {
	"name": "RAVDESS_AUs", //name of the task
	"data_type": "pathCSV",
	"model_name": "Sequence1Modal",
	"inference_data": "train", //Data to extract posteriors [Options: train, val]
	"OUT_PATH": "<RAVDESS_dir>/FUSION/wav2Vec_AUs/BiLSTM_AUS/posteriors", //Path to save the posteriors
	"TRAINED_PATH": "MMEmotionRecognition/src/Video/models/sequenceLearning/out/trained", //Path with the trained models
	"input_size": 35, 
	"model_params": // Parameters of the model saved in trained models 
	{"encoder":
		{
			"dim": 50,
			"layers": 2,
			"dropout": 0.3,
			"bidirectional": true

		},
	"attention":
		{
			"layers":2,
			"dropout": 0.3,
			"activation": "tanh",
			"context": false
		}
	},
	"preprocessor": null,
	"batch_size": 64,
	"lr": 1e-3,
	"weight_decay": 0.0,
	"patience": 30,
	"min_change": 0.0,
	"epochs": 300,
	"base": 0.0,
	"clip_norm": 1,
	"seed": 2020,
	"disable_cache": true
    }
