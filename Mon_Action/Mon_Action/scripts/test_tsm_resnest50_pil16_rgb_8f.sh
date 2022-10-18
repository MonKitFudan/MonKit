#!/usr/bin/env bash
python test_models.py pil16 \
    --weights=checkpoint/0-TSM_pil16_Flow_resnest50_shift8_blockres_avg_segment8_e25/ckpt.pth.tar \
    --test_segments=8 --batch_size=72 -j 24 --test_crops=3  --twice_sample  --full_res