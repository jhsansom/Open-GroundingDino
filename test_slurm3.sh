#!/bin/bash

#SBATCH --job-name=test_job_eecs545
#SBATCH --mail-user=jhsansom@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=30g
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
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

#PRETRAIN_MODEL_PATH=/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/model_weights/groundingdino_swint_ogc.pth
#PRETRAIN_MODEL_PATH=/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/model_weights/groundingdino_swinb_cogcoor.pth
#PRETRAIN_MODEL_PATH=/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/model_weights/groundingdino_swinb_cogcoor.pth.1
#PRETRAIN_MODEL_PATH=/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/model_checkpoints/checkpoint0002.pth
PRETRAIN_MODEL_PATH=/home/jhsansom/Open-GroundingDino/logs/checkpoint0002.pth
TEXT_ENCODER_PATH=bert-base-uncased

CFG=config/cfg_odvg.py
#DATASETS=config/datasets_od_test.json
DATASETS=config/refcoco_test.json
OUTPUT_DIR=./logs_test

python -u main.py --output_dir ${OUTPUT_DIR} \
    -c ${CFG} \
    --eval \
    --datasets ${DATASETS}  \
    --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
    --options text_encoder_type=${TEXT_ENCODER_PATH}