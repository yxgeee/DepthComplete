#!/bin/bash
python main.py --gpu-ids 6,7 -a sparsetodense \
	-b 32 --epochs 40 --step-size 8 --eval-step 1 --lr 0.01 --gamma 0.5 \
	--criterion masked_maeloss --tag sgd_bs32_new --optim sgd
