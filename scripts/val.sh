#!/bin/bash
python main.py --gpu-ids 0,1,2,3 -a sparseconv \
	--criterion masked_maeloss -b 1 --evaluate --resume ./checkpoints/kitti/sparseconv/masked_maeloss_adam/best_model.pth.tar
