#!/bin/bash
MODEL_PATH=$1
# ./checkpoints/kitti/sparseconv_masked_maeloss_adam

python test.py ${MODEL_PATH} --gpu-ids 4 -a sparseconv

./data/kitti/devkit/cpp/evaluate_depth ./data/kitti/depth_selection/val_selection_cropped/groundtruth_depth ${MODEL_PATH}/results
