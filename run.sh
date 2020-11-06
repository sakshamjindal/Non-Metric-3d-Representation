# python train.py --batch-size 5 \
#                 --seed 0 \
#                 --exp-dir single_obj_exp1 \
#                 --warmup-epoch 10 \
#                 --num-cluster 50 \
#                 --moco_r 40 \
#                 --hyp_N 1 \
#                 --mode "node" \
#                 --data "/home/mprabhud/dataset/clevr_lang/npys/aa_5t.txt"
                

python train.py --batch-size 5 \
                --seed 0 \
                --exp-dir single_obj_exp1 \
                --warmup-epoch 1 \
                --num-cluster 120 \
                --moco_r 40 \
                --hyp_N 1 \
                --mode "node" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/aa_5t.txt"