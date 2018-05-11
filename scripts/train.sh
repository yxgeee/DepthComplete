#!/bin/bash
python main.py --gpu-ids 0,1,2,3,4,5,6,7 \
	-b 64 --epochs 20 --step-size 5 --eval-step 5 --lr 0.1 \
	--save-root ./checkpoints/sparseconv