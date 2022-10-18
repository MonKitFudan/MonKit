
#python main.py pil14 RGB \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres

#python main.py hmdb51 Flow \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres


python main.py hmdb51 Flow \
     --arch resnest50_fast_4s2x40d --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
     --batch-size 72 -j 16 --dropout 0.8 --consensus_type=max --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres

#python main.py hmdb51 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 90 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres
#pil14 - flow
#python main.py pil14 Flow \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 90 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres
#
#python main.py pil14 Flow \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 90 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1

#python main.py pil14 Flow \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres

#python main.py pil14 Flow \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 90 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1

##hmdb51 - flow ...
#python main.py hmdb51 Flow \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres
#
#python main.py hmdb51 Flow \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1
#
#python main.py hmdb51 Flow \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres


###pil14
#
##rgb
#python main.py pil14 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 90 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres
#
##shift+rgb
#python main.py pil14 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 90 -j 16 --dropout 0.7 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres
#
#
##flow
#python main.py pil14 Flow \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.7 --consensus_type=avg --eval-freq=1
#
#resnest+rgb
#python main.py pil14 RGB \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 90 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1
#python main.py pil14 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 90 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1
##resnest+flow
#python main.py pil14 Flow \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.7 --consensus_type=avg --eval-freq=1
#
#
#
##shift+flow
#python main.py pil14 Flow \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.7 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres
#
##all +rgb
#python main.py pil14 RGB \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 90 -j 16 --dropout 0.7 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres
#
##all +flow
#python main.py pil14 Flow \
#     --arch resnest50_fast_4s2x40d --num_segments 8 \
#     --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
#     --batch-size 72 -j 16 --dropout 0.7 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres

