#!/bin/bash
python main.py --gpu-ids 0,1,2,3,4,5,6,7 \
	-b 64 --epochs 20 --step-size 0 --eval-step 5 --lr 0.001 \
	--save-root ./checkpoints/sparseconv