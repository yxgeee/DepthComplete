#!/bin/bash
python main.py --gpu-ids 3,4,5,6,7 -a pconvunet \
	-b 28 --epochs 20 --step-size 0 --eval-step 1 --lr 0.001 --gamma 0.5 \
	--criterion masked_maeloss --tag adam --optim adam