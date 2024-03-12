#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/ser_static_%A.out
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

### Feature Extraction from xlsr-Wav2Vec2.0

#### Extract embeddings
# To extract the features, first, we need to run the fine-tuning section to generate the train.csv and test.csv files. After running previous section, we could extract the features from the generated files, running the following command:

python3 ./src/Audio/FeatureExtractionWav2Vec/FeatureExtractor.py \
        --data ./data/models/wav2Vec_top_models/FineTuning/data/20211020_094500 \
        --model_id jonatasgrosman/wav2vec2-large-xlsr-53-english \
        --out_dir /data/FineTuningWav2Vec2_embs512

#### Train & Eval models

python3 ./src/Audio/FeatureExtractionWav2Vec/FeatureTraining.py \
            --embs_dir /data/embs512 \
            --model_number 11 \
            --param (80) \
            --type_of_norm 2 \
            --out_dir ./data/models/avg_MLP80_Audio