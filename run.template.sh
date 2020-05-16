#!/bin/sh
#SBATCH --account=def-jinguo
#SBATCH --job-name=de-simple
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=./logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kian.ahrabian@mail.mcgill.ca

source ${VENVDIR}/gg/bin/activate

python main.py -id ${SLURM_JOB_ID} \
    -dataset icews14 \
    -model DEDistMult \
    -s_dim 20 \
    -t_dim 64 \
    -r_dim 16 \
    -ne 500 \
    -bs 512 \
    -lr 0.001 \
    -nneg 500 \
    -drp 0.4 \
    -vd_stp 20 \
    -mtr mrr \
    -w 1 \
    -tr -vd -ts
