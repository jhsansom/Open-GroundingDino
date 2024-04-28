#!/bin/bash

#SBATCH --job-name=test_job_eecs545
#SBATCH --mail-user=jhsansom@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=30g
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=eecs545w24_class
#SBATCH --partition=spgpu
#SBATCH --output=./jobs/%u/%x-%j.log

module load cuda
source ../cleanrl-env/bin/activate

cd GroundingDINO
pip install -q -e .
cd ..

python eval.py