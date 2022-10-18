#!/usr/bin/env bash
python main.py pil14 RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
     --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1