#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=data_proc
#SBATCH --mail-user=zl3057@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=data_proc.out
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=n2c48m24


cd /scratch/$USER
singularity exec --nv\
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif\
    /bin/bash -c "source /ext3/env.sh; conda activate llama-finetune; cd /scratch/$USER/fine-tuning-climate-data; python3 data_processing.py"
