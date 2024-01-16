#!/bin/bash
#SBATCH --time= 10:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/ser_seq_%A.out
#SBATCH --job-name=ser
#SBATCH -n 1

# !kinit # to access data on /teamwork/
# !mkdir -p logs # to create ./logs/

module load miniconda
module load ffmpeg/4.3.2
module load gcc/8.4.0 
module load cuda

source activate mmer # created using requirement file: torchaudio version 0.10

## Speech Emotion Recognition

### Pre-processing: divided the audio into windows of 25 ms with an overlap of 15 ms and a stride of 20 ms
if false; then
python3 ./src/Audio/preProcessing/process_audio.py \
            --videos_dir /teamwork/t40511/emotion_av/RAVDESS/video_speech \
            --out_dir ./data/audios_16kHz
fi

### Fine-Tuning wav2vec2
#### training 27262008
python3 ./src/Audio/FineTuningWav2Vec/main_FineTuneWav2Vec_CV.py  \
            --audios_dir ./data/audios_16kHz --cache_dir ./data/Audio/cache_dir  \
            --out_dir ./data/FineTuningWav2Vec2_out  \
            --model_id jonatasgrosman/wav2vec2-large-xlsr-53-english

#### evaluation
# gp is here for fold in 0 1 2 3 4
if false; then
python3 ./src/Audio/FineTuningWav2Vec/Wav2VecEval.py \
             --data ./data/FineTuningWav2Vec2_out/data/20240115_143836 \
             --fold 0 \
             --trained_model ./data/FineTuningWav2Vec2_out/trained_models/wav2vec2-xlsr-ravdess-speech-emotion-recognition/20240115_143836 \
             --out_dir ./data/FineTuningWav2Vec2_posteriors  \
             --model_id jonatasgrosman/wav2vec2-large-xlsr-53-english

# ? wav2Vec_top_models
python3 ./src/Audio/FineTuningWav2Vec/Wav2VecEval.py \
             --data ./data/models/wav2Vec_top_models/FineTuning/data/20240115_143836 \
             --fold 0 \
             --trained_model ./data/models/wav2Vec_top_models/FineTuning/trained_models/wav2vec2-xlsr-ravdess-speech-emotion-recognition/20240115_143836 \
             --out_dir ./data/FineTuningWav2Vec2_posteriors \
             --model_id jonatasgrosman/wav2vec2-large-xlsr-53-english
# done

python3 ./src/Audio/FineTuningWav2Vec/FinalEvaluation.py \
             --dataPosteriors ./data/models/wav2Vec_top_models/FineTuning/posteriors/20211020_094500  \
             --trained_model ./data/models/wav2Vec_top_models/FineTuning/trained_models/wav2vec2-xlsr-ravdess-speech-emotion-recognition/20211020_094500
fi