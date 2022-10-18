#!/usr/bin/env bash

#python main.py pil14 RGB \
#     --arch resnest101abalation --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 56 -j 16 --dropout 0.5 --consensus_type=wavg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres

python main.py pil14 Flow \
     --arch resnest101abalation --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
     --batch-size 64 -j 16 --dropout 0.5 --consensus_type=wavg --eval-freq=1