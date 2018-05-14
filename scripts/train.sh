#!/bin/bash
python main.py --gpu-ids 5 -a sparsetodense \
	-b 16 --epochs 20 --step-size 5 --eval-step 1 --lr 0.001 --gamma 0.5 \
	--criterion masked_maeloss --tag adam --optim adam