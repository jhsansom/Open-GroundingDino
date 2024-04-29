#!/bin/bash

#SBATCH --job-name=test_job_eecs545
#SBATCH --mail-user=jhsansom@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=30g
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=eecs545w24_class
#SBATCH --partition=spgpu
#SBATCH --output=./jobs/%u/%x-%j.log

module load cuda
source ../cleanrl-env/bin/activate

cd GroundingDINO
pip install -q -e .
cd ..

# Epoch -1 (prior to any training)
WEIGHTS="/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/model_weights/gdinot-1.8m-odvg.pth"
python eval.py --real=True --spatial=True --weights=$WEIGHTS # refcoco spatial
python eval.py --real=True --spatial=False --weights=$WEIGHTS # refcoco nonspatial
python eval.py --real=False --weights=$WEIGHTS # synthetic validation

# Epoch 0 (after 1 epoch of training)
WEIGHTS="./logs/checkpoint0000.pth"
python eval.py --real=True --spatial=True --weights=$WEIGHTS # refcoco spatial
python eval.py --real=True --spatial=False --weights=$WEIGHTS # refcoco nonspatial
python eval.py --real=False --weights=$WEIGHTS # synthetic validation

# Epoch 1 (after 2 epochs of training)
WEIGHTS="./logs/checkpoint0001.pth"
python eval.py --real=True --spatial=True --weights=$WEIGHTS # refcoco spatial
python eval.py --real=True --spatial=False --weights=$WEIGHTS # refcoco nonspatial
python eval.py --real=False --weights=$WEIGHTS # synthetic validation

# Epoch 2 (after 3 epochs of training)
WEIGHTS="./logs/checkpoint0002.pth"
python eval.py --real=True --spatial=True --weights=$WEIGHTS # refcoco spatial
python eval.py --real=True --spatial=False --weights=$WEIGHTS # refcoco nonspatial
python eval.py --real=False --weights=$WEIGHTS # synthetic validation

# Epoch 3 (after 4 epochs of training)
WEIGHTS="./logs/checkpoint0003.pth"
python eval.py --real=True --spatial=True --weights=$WEIGHTS # refcoco spatial
python eval.py --real=True --spatial=False --weights=$WEIGHTS # refcoco nonspatial
python eval.py --real=False --weights=$WEIGHTS # synthetic validation

# Epoch 4 (after 5 epochs of training)
WEIGHTS="./logs/checkpoint0004.pth"
python eval.py --real=True --spatial=True --weights=$WEIGHTS # refcoco spatial
python eval.py --real=True --spatial=False --weights=$WEIGHTS # refcoco nonspatial
python eval.py --real=False --weights=$WEIGHTS # synthetic validation

# Epoch 5 (after 6 epochs of training)
WEIGHTS="./logs/checkpoint0005.pth"
python eval.py --real=True --spatial=True --weights=$WEIGHTS # refcoco spatial
python eval.py --real=True --spatial=False --weights=$WEIGHTS # refcoco nonspatial
python eval.py --real=False --weights=$WEIGHTS # synthetic validation

# Epoch 6 (after 7 epochs of training)
WEIGHTS="./logs/checkpoint0006.pth"
python eval.py --real=True --spatial=True --weights=$WEIGHTS # refcoco spatial
python eval.py --real=True --spatial=False --weights=$WEIGHTS # refcoco nonspatial
python eval.py --real=False --weights=$WEIGHTS # synthetic validation