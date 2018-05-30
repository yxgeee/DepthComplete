#!/bin/bash
python main.py --gpu-ids 0,1,2,3 -a sparseconv \
	-b 64 --epochs 40 --step-size 20 --eval-step 1 --lr 0.001 --gamma 0.5 \
	--criterion masked_maeloss --tag adam --optim adam
