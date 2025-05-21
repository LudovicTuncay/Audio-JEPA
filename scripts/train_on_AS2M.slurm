#!/bin/sh
# -*- coding: utf-8 -*-

# ------- SLURM PARAMETERS -------------
# Account
#SBATCH --account=ojz@v100

# Quality of service

#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00                                     # maximum execution time requested (HH:MM:SS)

#SBATCH --nodes=1                                           # requesting one node
#SBATCH --ntasks-per-node=4                                 # with one task per node (= number of GPUs here)

#SBATCH --gres=gpu:4                                        # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=10                                  # number of CPUs per task (1/4 of the CPUs of the 4-GPU V100 node)

#SBATCH --hint=nomultithread                                # hyperthreading disabled
#SBATCH --output=logs/JZ_sbatch_logs/AudioJEPA-%j.out       # output file name
#SBATCH --error=logs/JZ_sbatch_logs/AudioJEPA-%j.out        # error file name (here, same as output)



# --------- PROGRAM START ------------
set -x

cd ${SLURM_SUBMIT_DIR}

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TMPDIR=$SCRATCH
export TEMP=$SCRATCH
export TMP=$SCRATCH
export COLUMNS=200

source .venv/bin/activate

# Code execution
srun .venv/bin/python -u -O src/train.py paths=default_linux data=audioset_V100_unbal trainer=ddp trainer.devices=4 logger=wandb logger.wandb.offline=True trainer.max_steps=100000 callbacks.model_checkpoint.save_weights_only=true data.batch_size=64
