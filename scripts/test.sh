#!/bin/bash
MODEL_PATH=$1
GPU_ID=$2
MODEL_NAME=$3

python test.py ${MODEL_PATH} --gpu-ids ${GPU_ID} -a ${MODEL_NAME}

./data/kitti/devkit/cpp/evaluate_depth ./data/kitti/depth_selection/val_selection_cropped/groundtruth_depth ${MODEL_PATH}/results
