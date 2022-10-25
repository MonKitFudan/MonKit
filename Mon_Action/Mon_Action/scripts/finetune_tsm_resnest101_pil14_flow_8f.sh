#!/usr/bin/env bash
python main.py pil13 Flow \
     --arch resnet101 --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
     --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres \
     --tune_from=pretrained/TSM_something_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth