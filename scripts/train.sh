#!/bin/bash
python main.py --gpu-ids 0,1,2,3,4,5,6,7 -a pconvunet \
	-b 32 --epochs 20 --step-size 0 --eval-step 1 --lr 0.001 --gamma 0.5 \
	--criterion masked_maeloss --tag adam_nobn --optim adam
