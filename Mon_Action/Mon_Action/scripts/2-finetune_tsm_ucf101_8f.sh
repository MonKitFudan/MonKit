#python main.py pil14 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1



python main.py hmdb51 Flow \
     --arch resnest50 --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
     --batch-size 72 -j 16 --dropout 0.8 --consensus_type=wavg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres