#!/usr/bin/env bash
python test_models.py pil14 \
    --weights=checkpoint/TSM_pil14_RGB_resnest50abalation_shift8_blockres_avg_segment8_e50/ckpt.pth.tar \
    --test_segments=8 --batch_size=64 -j 24 --test_crops=3  --twice_sample  --full_res