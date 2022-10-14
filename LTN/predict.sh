#!/bin/bash
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=00:20:00
#PBS -l wd
#PBS -l storage=scratch/po21
#PBS -m abe
#PBS -N temp
#PBS -l jobfs=100GB

module load cuda/10.1
module load cudnn/7.6.5-cuda10.1 
source /home/549/jz1585/.miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate torch


cd /scratch/po21/jz1585/Label_noise/joint/Temp/RAN_CRF/
# python train.py --config_path config2.json
python predict.py --config_path pre_config.json
