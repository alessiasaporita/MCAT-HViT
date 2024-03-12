#!/bin/bash
#SBATCH --job-name=MCAT_main
#SBATCH --output=/work/ai4bio2023/MCAT-HViT/Output/Test/output_test
#SBATCH --error=/work/ai4bio2023/MCAT-HViT/Output/Error/output_error
#SBATCH --gres=gpu:1
#SBATCH --account=ai4bio2023
#SBATCH --time=24:00:00
#SBATCH --mem=30GB
#SBATCH --partition=all_usr_prod

cd /work/ai4bio2023/MCAT-HViT
source activate /homes/asaporita/.conda/envs/MCAT


python -u main.py 