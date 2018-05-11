#!/bin/bash
python main.py --gpu-ids 0,1,2,3,4,5,6,7 \
	-b 80 --epochs 20 --step-size 0 --eval-step 1 --lr 0.001 \
	--criterion masked_mseloss --tag adam