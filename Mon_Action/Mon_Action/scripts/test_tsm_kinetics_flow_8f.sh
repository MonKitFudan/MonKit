# test TSM
python genFlow.py kinetics \
    --weights=pretrained/TSM_kinetics_Flow_resnet50_shift8_blockres_avg_segment8_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64