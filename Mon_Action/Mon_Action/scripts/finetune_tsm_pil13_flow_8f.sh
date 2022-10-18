python main.py pil13 RGB \
    --arch resnest101 --num_segments 8 \
    --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
    --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres --dense_sample

python main.py pil13 Flow \
    --arch resnest101 --num_segments 8 \
    --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
    --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres --dense_sample