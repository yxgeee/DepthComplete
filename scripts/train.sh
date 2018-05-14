#!/bin/bash
python main.py --gpu-ids 4,5,6,7 -a sparsetodense \
	-b 64 --epochs 20 --step-size 10 --eval-step 1 --lr 0.003 --gamma 0.5 \
	--criterion masked_maeloss --tag adam --optim adam
