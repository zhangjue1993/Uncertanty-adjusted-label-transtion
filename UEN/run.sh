#!/bin/bash
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=01:00:00
#PBS -l wd
#PBS -l storage=scratch/po21
#PBS -m abe
#PBS -N temp
#PBS -l jobfs=100GB

module load cuda/10.1
module load cudnn/7.6.5-cuda10.1 
conda init bash
conda activate torch



python train.py --config_path act_config.json
#python predict.py --config_path act_config.json
