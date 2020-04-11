#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J finetuneb64
#BSUB -n 1
#BSUB -W 12:00
#BSUB -R "rusage[mem=14GB]"
#BSUB -u peter_ebert@live.dk
#BSUB -o %J.out
#BSUB -e %J.err

# Load modules
nvidia-smi
module load python3/3.6.2
module load cuda/10.2
#module load cudnn/v7.0-prod-cuda8

# Execute script
cd /zhome/3f/6/108837/Desktop/program_synthesis/program_synthesis/
python3 train_rl.py --batch_size 64 --lr 1e-6 --lr_decay_rate 0.99 --max_rollout_length 1
#--uuid default \
#        --domain walker --task walk \
#        --encoder_type pixel --use_inv 1