#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=llama_finetune
#SBATCH --mail-user=zl3057@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=llama_finetune.out
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=n1s8-v100-1


cd /scratch/$USER
singularity exec --nv \
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate llama-finetune
        cd /scratch/$USER/fine-tuning-climate-data
        python3 fine_tuning.py
    "
