#!/bin/bash
python test.py ./checkpoints/kitti/sparseconv_masked_maeloss_adam/best_model.pth.tar \
	--gpu-ids 4 -a sparseconv --tag test_sparseconv_mae
