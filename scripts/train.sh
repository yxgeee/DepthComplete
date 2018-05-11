#!/bin/bash
python main.py --gpu-ids 0,1,2,3 \
	-b 32 --epochs 20 --step-size 0 --eval-step 1 --lr 0.001 \
	--criterion masked_mseloss --tag adam