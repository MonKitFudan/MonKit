#!/usr/bin/env bash
python test_models.py ucf101 \
    --weights=checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.pth.tar \
    --test_segments=8 --batch_size=18 -j 24 --test_crops=3  --twice_sample  --full_res