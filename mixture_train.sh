#!/bin/bash

#SBATCH --job-name=test_job_eecs545
#SBATCH --mail-user=jhsansom@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=50g
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --output=./jobs/%u/%x-%j.log

module load cuda
source ../cleanrl-env/bin/activate
cd models/GroundingDINO/ops

#echo "Installing library"
#python setup.py build install

#echo "Running test"
#python test.py
cd ../../..

PRETRAIN_MODEL_PATH=/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/model_weights/gdinot-1.8m-odvg.pth
TEXT_ENCODER_PATH=bert-base-uncased

CFG=config/cfg_odvg.py
DATASETS=config/mixture_train.json
OUTPUT_DIR=./logs

# Watch this line: it's dangerous
rm -rf ./logs/

python -u main.py --output_dir ${OUTPUT_DIR} \
    -c ${CFG} \
    --datasets ${DATASETS}  \
    --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
    --options text_encoder_type=${TEXT_ENCODER_PATH} \
