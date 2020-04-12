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
python3 train.py --dataset karel --model_type karel-lgrl-ref --karel-mutate-ref --karel-mutate-n-dist 1,2,3 --karel-trace-enc none --karel-refine-dec edit --num_placeholders 0 --debug_every_n=1000 --eval_every_n=10000000 --keep_every_n=10000 --log_interval=100 --batch_size 64 --num_epochs 50 --max_beam_trees 1 --optimizer sgd --gradient-clip 1 --lr 1 --lr_decay_steps 100000 --lr_decay_rate 0.5 --model_dir logdirs/vanilla,trace_enc==none,batch_size==64,lr==1,lr_decay_steps=100000
#python3 train.py --dataset karel --model_type karel-lgrl-ref --karel-mutate-ref --karel-mutate-n-dist 1,2,3 --karel-trace-enc aggregate:conv_all_grids=True,rnn_trace=True --karel-refine-dec edit --num_placeholders 0 --debug_every_n=1000 --eval_every_n=10000000 --keep_every_n=10000 --log_interval=100 --batch_size 64 --num_epochs 50 --max_beam_trees 1 --optimizer sgd --gradient-clip 1 --lr 1 --lr_decay_steps 100000 --lr_decay_rate 0.5 --model_dir logdirs/aggregate-with-io-rnn-traces,trace_enc==aggregate:conv_all_grids=True,rnn_trace=True,batch_size==64,lr==1,lr_decay_steps=100000
#python3 train_rl.py --batch_size 64 --lr 1e-6 --lr_decay_rate 0.99 --max_rollout_length 1
#--uuid default \
#        --domain walker --task walk \
#        --encoder_type pixel --use_inv 1