#!/bin/bash
python main.py --gpu-ids 4,5,6,7 \
	-b 32 --epochs 20 --step-size 0 --eval-step 1 --lr 0.001 \
	--criterion masked_maeloss --tag adam