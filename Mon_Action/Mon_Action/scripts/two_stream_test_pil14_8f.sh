#!/usr/bin/env bash
python test_models.py pil14 \
    --weights=checkpoint/TSM_pil14_Flow_resnest50_shift8_blockres_avg_segment8_e50_dense/ckpt.pth.tar \
    --test_segments=8 --batch_size=16 -j 24 --test_crops=3  --twice_sample --crop_fusion_type=avg